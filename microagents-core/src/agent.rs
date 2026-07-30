use std::{
    collections::HashMap,
    fmt::{self, Debug},
    fs, io,
    path::{Path, PathBuf},
    str::FromStr,
    sync::Arc,
};

use async_stream::stream;
use chrono::Utc;
use futures_util::StreamExt;
use llms_sdk::{
    ApiType, LLM, LLMRequest, LLMStreamingResponse, Message, MessagePart, MessageRole, TextPart,
    Tool, ToolCallPart, ToolResultPart,
};
use microagents_events::{
    AgentEventAny, AssistantMessagePart, AssistantResponseEvent, DeltaType, SessionInitEvent,
    SessionInitType, SessionStopEvent, SkillLoadEvent, StreamDeltaEvent, TaskEvent, TaskStatus,
    TextPart as AssistantTextPart, ThinkingPart as AssistantThinkingPart, ToolCallAnyEvent,
    ToolCallEvent, ToolCallPart as AssistantToolCallPart, ToolResultEvent, Usage,
    UserPromptSubmitEvent, types::ToolResult,
};
use microagents_storage::{
    jsonl::JsonlAgentStorage,
    memory::InMemoryAgentStorage,
    sqlite::SqliteAgentStorage,
    types::{AgentStorage, AgentStorageChoice},
};
use serde_json::{Value, json};
use thiserror::Error;
use tokio::{
    sync::{Mutex, Semaphore},
    task::JoinSet,
};

use crate::{
    common::{
        call_tool, check_env_var, convert_event_to_message, get_incomplete_tasks, load_agents_md,
    },
    skills::{self, ensure_skill_with_project_path, find_skills_with_project_path, parse_skill},
    types::{Agent, AgentError, GenerationStream, RunStream, ToolExecutionContext, ToolFunction},
};

/// Relative path to the default project-local skills directory.
pub use crate::skills::SKILLS_PATH;
/// Path alias for the default global skills directory (resolved at runtime).
pub const GLOBAL_SKILLS_PATH: &str = "~/.agents/skills";
/// Name of the built-in skill-loading tool exposed to the LLM.
pub const SKILLS_TOOL_NAME: &str = "skills";
/// Name of the built-in task tracking tool exposed to the LLM.
pub const TASKS_TOOL_NAME: &str = "tasks";
/// Base system prompt injected into every conversation.
pub const BASE_SYSTEM_PROMPT: &str = r#"<identity>
You are MicroAgent, an AI agent whose purpose is to
fulfil request coming from a user, employing the tools and skills
available to you and interacting with the environment
you are given
</identity>
<guidelines>
<general>
To carry out a task, follow the main rules of the Zen of Python whenever possible:
- Beautiful is better than ugly.
- Explicit is better than implicit.
- Simple is better than complex.
- Complex is better than complicated.
- Flat is better than nested.
- Readability counts.
- Special cases aren't special enough to break the rules, although practicality beats purity.
- Errors should never pass silently, unless explicitly silenced.
- In the face of ambiguity, refuse the temptation to guess.
- There should be one (and preferably only one) obvious way to do it.
- If the implementation is hard to explain, it's a bad idea.
- If the implementation is easy to explain, it _may_ be a good idea, but **it is not necessarily**.
</general>
<tools_and_skills_usage>
Tools can be invoked by providing their name and an input conforming to their input JSON schema.
Call tools either when requested by the user, or when the description of the tool seems compelling
enough for the task at hand.
You also have a special tool called 'skills'. When you want to access specialized knowledge over a
particular area, you can invoke the skill pertaining to that area by calling the 'skills' tool and
providing the name of the skill to it. The 'skills' tool will return the specific instructions for that
skill. Invoke a skill either when directly prompted by the user to do so, or when the skill's description
seems compelling enough for the task at hand.
</tools_and_skills_usage>
</guidelines>
"#;
/// Maximum number of tool calls executed concurrently when
/// `parallel_tool_calls` is enabled.
const MAX_CONCURRENT_TOOL_CALLS: usize = 10;

async fn persist_event(
    storage: &dyn AgentStorage,
    event: &AgentEventAny,
) -> Result<(), AgentError> {
    storage
        .update_session(event.clone())
        .await
        .map_err(|error| {
            AgentError::RunError(format!(
                "An error occurred while updating the session in the storage: {error}"
            ))
        })
}

fn assistant_message_parts(message: &Message) -> Result<Vec<AssistantMessagePart>, AgentError> {
    message
        .content
        .iter()
        .map(|part| match part {
            MessagePart::Text(text) => Ok(AssistantMessagePart::Text(AssistantTextPart {
                text: text.text.clone(),
            })),
            MessagePart::Thinking(thinking) => {
                Ok(AssistantMessagePart::Thinking(AssistantThinkingPart {
                    thinking: thinking.thinking.clone(),
                    signature: thinking.signature.clone(),
                }))
            }
            MessagePart::ToolCall(tool_call) => {
                Ok(AssistantMessagePart::ToolCall(AssistantToolCallPart {
                    id: tool_call.id.clone(),
                    name: tool_call.name.clone(),
                    arguments: tool_call.arguments.clone(),
                }))
            }
            _ => Err(AgentError::RunError(
                "Assistant response contains an unsupported message part".to_string(),
            )),
        })
        .collect()
}

/// Supported LLM providers.
#[derive(Debug, Hash, PartialEq, Eq, Clone, Default)]
pub enum SupportedProvider {
    #[default]
    OpenAI,
    Anthropic,
}

impl From<SupportedProvider> for ApiType {
    fn from(val: SupportedProvider) -> Self {
        match val {
            SupportedProvider::OpenAI => ApiType::OpenAI,
            SupportedProvider::Anthropic => ApiType::Anthropic,
        }
    }
}

pub trait AsProvider {
    fn as_provider(&self) -> Result<SupportedProvider, MicroAgentBuilderError>;
}

impl AsProvider for SupportedProvider {
    fn as_provider(&self) -> Result<SupportedProvider, MicroAgentBuilderError> {
        Ok(self.to_owned())
    }
}

impl FromStr for SupportedProvider {
    type Err = MicroAgentBuilderError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "openai" => Ok(Self::OpenAI),
            "anthropic" => Ok(Self::Anthropic),
            _ => Err(MicroAgentBuilderError::ProviderNotSupported(s.into())),
        }
    }
}

impl AsProvider for String {
    fn as_provider(&self) -> Result<SupportedProvider, MicroAgentBuilderError> {
        SupportedProvider::from_str(self)
    }
}

impl fmt::Display for SupportedProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::OpenAI => "openai",
            Self::Anthropic => "anthropic",
        };
        write!(f, "{}", s)
    }
}

impl SupportedProvider {
    /// Return the default model identifier for this provider.
    pub fn default_model(&self) -> Result<&'static str, MicroAgentBuilderError> {
        match self {
            // GPT-5.5 is the current default ChatGPT model as of May 2026
            SupportedProvider::OpenAI => Ok("gpt-5.6-terra"),

            // Claude Opus 4.7 by Anthropic is cuttig-edge in the models market
            SupportedProvider::Anthropic => Ok("claude-sonnet-5"),
        }
    }
}

/// Errors that can occur while configuring or building a [`MicroAgent`].
#[derive(Debug, Error)]
pub enum MicroAgentBuilderError {
    #[error("Skill {0} not found")]
    SkillNotFound(String),
    #[error("Invalid skill name {0}")]
    InvalidSkillName(String),
    #[error("Skill parsing error")]
    SkillParsingError(#[from] skills::SkillLoadingError),
    #[error("Provider {0} not supported")]
    ProviderNotSupported(String),
    #[error("Tool with name {0} already exists")]
    ToolAlreadyDefined(String),
    #[error("Storage could not be loaded: {0}")]
    StorageLoadError(String),
    #[error("Environment variable {0} not found")]
    EnvVarNotFoundError(String),
    #[error("Provider {0} should specify a model")]
    ModelNotSpecifiedError(String),
    #[error(transparent)]
    AgentsMdResolutionError(#[from] io::Error),
}

/// A fully-configured agent ready to generate responses or run conversations.
///
/// Created via [`MicroAgentBuilder`]. Holds the conversation history, tool
/// registry, and LLM client configuration.
pub struct MicroAgent<Ctx> {
    pub history: Vec<Message>,
    pub tools: HashMap<String, Arc<dyn ToolFunction<Ctx>>>,
    pub skills: HashMap<String, String>,
    skills_path: PathBuf,
    pub provider: SupportedProvider,
    pub base_url: Option<String>,
    pub model: String,
    pub system: String,
    pub api_key: String,
    client: Arc<LLM>,
    pub tool_context: Arc<ToolExecutionContext<Ctx>>,
    pub storage: Box<dyn AgentStorage>,
    pub parallel_tool_calls: bool,
    pub tasks: Arc<Mutex<HashMap<String, TaskStatus>>>,
    pub prompt_cache: bool,
}

impl<Ctx: Debug> Debug for MicroAgent<Ctx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MicroAgent")
            .field("history", &self.history)
            .field("base_url", &self.base_url)
            .field("tools", &self.tools)
            .field("skills", &self.skills)
            .field("skills_path", &self.skills_path)
            .field("provider", &self.provider)
            .field("model", &self.model)
            .field("system", &self.system)
            .field("client", &self.client)
            .field("tool_context", &self.tool_context)
            .field("storage", &self.storage)
            .field("parallel_tool_calls", &self.parallel_tool_calls)
            .finish()
    }
}

/// Builder for [`MicroAgent`].
///
/// # Example
/// ```no_run
/// use microagents_core::agent::MicroAgentBuilder;
/// use microagents_core::types::ToolExecutionContext;
///
/// let agent = MicroAgentBuilder::new(ToolExecutionContext::new(()))
///     .provider("openai".to_string()).unwrap()
///     .model("gpt-5.5")
///     .build()
///     .expect("API key must be set");
/// ```
#[derive(Debug)]
pub struct MicroAgentBuilder<Ctx> {
    tools: HashMap<String, Arc<dyn ToolFunction<Ctx>>>,
    skills: HashMap<String, String>,
    provider: SupportedProvider,
    model: String,
    custom_instructions: String,
    skills_path: PathBuf,
    tool_context: Arc<ToolExecutionContext<Ctx>>,
    base_url: Option<String>,
    prompt_cache: bool,
    pub storage: Box<dyn AgentStorage>,
    pub parallel_tool_calls: bool,
}

impl<Ctx: Send + Sync + 'static> MicroAgentBuilder<Ctx> {
    /// Create a new builder with the given tool execution context.
    ///
    /// The `skills` tool is registered automatically.
    pub fn new(tool_context: ToolExecutionContext<Ctx>) -> Self {
        Self {
            tools: HashMap::from([
                (
                    SKILLS_TOOL_NAME.to_string(),
                    Arc::new(SkillsTool) as Arc<dyn ToolFunction<Ctx>>,
                ),
                (
                    TASKS_TOOL_NAME.to_string(),
                    Arc::new(TasksTool) as Arc<dyn ToolFunction<Ctx>>,
                ),
            ]),
            skills: HashMap::new(),
            provider: SupportedProvider::default(),
            model: String::new(),
            custom_instructions: String::new(),
            skills_path: PathBuf::from(SKILLS_PATH),
            tool_context: Arc::new(tool_context),
            storage: Box::new(InMemoryAgentStorage::default()) as Box<dyn AgentStorage>,
            parallel_tool_calls: false,
            base_url: None,
            prompt_cache: true,
        }
    }

    /// Register a single skill by name.
    ///
    /// Searches the configured project skills directory first, then
    /// `~/.agents/skills/{name}`.
    ///
    /// Call [`Self::skills_path`] before this method when using a non-default
    /// project skills directory. Registered skills are not re-resolved if the
    /// path changes later.
    pub fn add_skill(
        mut self,
        skill_name: impl Into<String>,
    ) -> Result<Self, MicroAgentBuilderError> {
        let skill_name = skill_name.into();
        if !skills::is_valid_skill_name(&skill_name) {
            return Err(MicroAgentBuilderError::InvalidSkillName(skill_name));
        }
        if let Some(skill_path) = ensure_skill_with_project_path(&self.skills_path, &skill_name) {
            let description = parse_skill(&skill_path.join("SKILL.md"))?;
            self.skills.insert(skill_name, description);
            return Ok(self);
        }
        Err(MicroAgentBuilderError::SkillNotFound(skill_name))
    }

    /// Auto-discover and register all skills found in the configured project
    /// and global skills directories.
    ///
    /// Call [`Self::skills_path`] before this method when using a non-default
    /// project skills directory. Discovered skills are not re-resolved if the
    /// path changes later.
    pub fn find_skills(mut self) -> Result<Self, MicroAgentBuilderError> {
        let loaded_skills = find_skills_with_project_path(&self.skills_path)?;
        for (skill, des) in loaded_skills {
            self.skills.insert(skill, des);
        }
        Ok(self)
    }

    /// Set the project-local skills directory.
    ///
    /// This directory takes precedence over `~/.agents/skills` and replaces
    /// the default `.agents/skills` lookup directory for this agent.
    pub fn skills_path(mut self, skills_path: impl Into<PathBuf>) -> Self {
        self.skills_path = skills_path.into();
        self.tools.insert(
            SKILLS_TOOL_NAME.to_string(),
            Arc::new(ConfiguredSkillsTool {
                skills_path: self.skills_path.clone(),
            }) as Arc<dyn ToolFunction<Ctx>>,
        );
        self
    }

    /// Set the LLM provider (`"openai"` or `"anthropic"`).
    pub fn provider(mut self, provider: impl AsProvider) -> Result<Self, MicroAgentBuilderError> {
        let prov = provider.as_provider()?;
        self.provider = prov;
        Ok(self)
    }

    /// Set the base URL for the LLM provider. If unset, attempts to read from environment variables and,
    /// if those are unset too, falls back to the default API url for the chosen provider
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Enable or disable prompt caching (for Anthropic, OpenAI always uses it).
    ///
    /// Enabled by default.
    pub fn prompt_cache(mut self, prompt_cache: bool) -> Self {
        self.prompt_cache = prompt_cache;
        self
    }

    /// Set the model identifier. If empty, the provider's default is used.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Enable or disable parallel tool execution.
    pub fn parallel_tool_calls(mut self, parallel_tool_calls: bool) -> Self {
        self.parallel_tool_calls = parallel_tool_calls;
        self
    }

    /// Configure the session storage backend.
    pub async fn storage(
        mut self,
        storage: AgentStorageChoice,
    ) -> Result<Self, MicroAgentBuilderError> {
        match storage {
            AgentStorageChoice::Jsonl => self.storage = Box::new(JsonlAgentStorage::default()),
            AgentStorageChoice::Memory => self.storage = Box::new(InMemoryAgentStorage::default()),
            AgentStorageChoice::Sqlite => {
                let store = SqliteAgentStorage::new(None)
                    .await
                    .map_err(|e| MicroAgentBuilderError::StorageLoadError(e.to_string()))?;
                self.storage = Box::new(store);
            }
        }

        Ok(self)
    }

    /// Register a custom tool.
    pub fn add_tool(
        mut self,
        tool: Arc<dyn ToolFunction<Ctx>>,
    ) -> Result<Self, MicroAgentBuilderError> {
        let name = tool.name();
        if self.tools.contains_key(name) {
            return Err(MicroAgentBuilderError::ToolAlreadyDefined(name.to_owned()));
        }
        self.tools.insert(name.to_owned(), tool);
        Ok(self)
    }

    /// Append free-form instructions to the system prompt.
    pub fn custom_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.custom_instructions = instructions.into();
        self
    }

    /// Load custom instructions from an AGENTS.md file in the current repository
    pub fn load_agents_md(mut self) -> Result<Self, MicroAgentBuilderError> {
        let instructions = load_agents_md()?;
        if let Some(instr) = instructions {
            self.custom_instructions += &instr;
        }
        Ok(self)
    }

    /// Choose the effective model: user-supplied or provider default.
    fn resolve_model(&self) -> Result<String, MicroAgentBuilderError> {
        if self.model.is_empty() {
            return self.provider.default_model().map(|m| m.to_string());
        }
        Ok(self.model.clone())
    }

    /// Assemble the full system prompt from the base prompt, model info,
    /// registered tools, skills, and any custom instructions.
    fn resolve_system(&self, model: &str) -> String {
        let mut base = BASE_SYSTEM_PROMPT.to_string();
        base += &format!(
            r#"<model>
You are {} provided by {}
</model>
"#,
            model, self.provider
        );
        if !self.tools.is_empty() {
            base += "\n<tools>";
            for (k, v) in &self.tools {
                base += &format!(
                    "\n<tool>\n<name>{}</name>\n<description>{}</description>\n<input_schema>{}</input_schema>\n</tool>",
                    k,
                    v.description(),
                    v.input_schema()
                )
            }
            base += "\n</tools>"
        }
        if !self.skills.is_empty() {
            base += "\n<skills>";
            for (k, v) in &self.skills {
                base += &format!(
                    "\n<skill>\n<name>{}</name>\n<description>{}</description>\n</skill>",
                    k, v
                );
            }
            base += "\n</skills>";
        }
        if !self.custom_instructions.is_empty() {
            base += &format!(
                "\n<additional_instructions>\n{}\n</additional_instructions>",
                self.custom_instructions
            )
        }

        base
    }

    fn resolve_base_url(&self) -> Option<String> {
        if let Some(url) = self.base_url.clone() {
            Some(url)
        } else {
            match self.provider {
                SupportedProvider::OpenAI => match check_env_var("OPENAI_BASE_URL") {
                    Ok(u) => Some(u),
                    Err(_) => Some("https://api.openai.com/v1".to_string()),
                },
                SupportedProvider::Anthropic => match check_env_var("ANTHROPIC_BASE_URL") {
                    Ok(u) => Some(u),
                    Err(_) => Some("https://api.anthropic.com/v1".to_string()),
                },
            }
        }
    }

    /// Finalise the builder and return a [`MicroAgent`].
    ///
    /// Fails early if a required API key is missing for the chosen provider.
    #[must_use = "The builder needs to call `build` otherwise it hangs without turning into an actual agent."]
    pub fn build(self) -> Result<MicroAgent<Ctx>, MicroAgentBuilderError> {
        let model = self.resolve_model()?;
        let system = self.resolve_system(&model);
        let api_key = match self.provider {
            SupportedProvider::OpenAI => check_env_var("OPENAI_API_KEY").map_err(|_| {
                MicroAgentBuilderError::EnvVarNotFoundError("OPENAI_API_KEY".into())
            })?,
            SupportedProvider::Anthropic => check_env_var("ANTHROPIC_API_KEY").map_err(|_| {
                MicroAgentBuilderError::EnvVarNotFoundError("ANTHROPIC_API_KEY".into())
            })?,
        };
        Ok(MicroAgent {
            history: vec![],
            base_url: self.resolve_base_url(),
            tools: self.tools,
            skills: self.skills,
            skills_path: self.skills_path,
            model,
            api_key,
            provider: self.provider,
            client: Arc::new(LLM::default()),
            system,
            tool_context: self.tool_context,
            storage: self.storage,
            parallel_tool_calls: self.parallel_tool_calls,
            tasks: Arc::new(Mutex::new(HashMap::new())),
            prompt_cache: self.prompt_cache,
        })
    }
}

impl<Ctx> MicroAgent<Ctx> {
    pub fn resolve_prompt(&self, prompt: &str) -> Result<String, AgentError> {
        if prompt.starts_with("/") {
            let parts: Vec<&str> = prompt.split_whitespace().collect();
            let skill = parts
                .first()
                .map(|s| s.trim_start_matches("/"))
                .unwrap_or("");
            if self.skills.contains_key(skill) {
                let skill_path = ensure_skill_with_project_path(&self.skills_path, skill);
                match skill_path {
                    Some(p) => {
                        let content = fs::read_to_string(p.join("SKILL.md"))
                            .map_err(|_| AgentError::SkillResolutionError)?;
                        return Ok(prompt.replace(
                            &format!("/{}", skill),
                            &format!("<skill>\n{}\n</skill>\n\n", content),
                        ));
                    }
                    None => return Err(AgentError::SkillResolutionError),
                }
            }
        }
        Ok(prompt.to_owned())
    }
}

/// Built-in tool that loads skill instructions at runtime.
#[derive(Debug)]
pub struct SkillsTool;

#[derive(Debug)]
struct ConfiguredSkillsTool {
    skills_path: PathBuf,
}

/// Built-in tool that track agent tasks
#[derive(Debug)]
pub struct TasksTool;

fn load_skill(skill_name: &str, skills_path: &Path) -> Result<ToolResult, AgentError> {
    if !skills::is_valid_skill_name(skill_name) {
        return Ok(ToolResult::Err(format!(
            "Skill name {skill_name:?} is invalid"
        )));
    }
    if let Some(path) = ensure_skill_with_project_path(skills_path, skill_name) {
        let content = fs::read_to_string(path.join("SKILL.md")).map_err(|error| {
            AgentError::ToolCallError(format!("Skill {skill_name} could not be read: {error}"))
        })?;
        return Ok(ToolResult::Ok(content));
    }
    Ok(ToolResult::Err(format!(
        "Skill {skill_name} could not be found"
    )))
}

fn skills_tool_input_schema() -> Value {
    json!({
      "type": "object",
      "required": [
        "skill_name"
      ],
      "properties": {
        "skill_name": {
          "type": "string",
          "description": "Name of the skill to load"
        }
      }
    })
}

#[async_trait::async_trait]
impl<Ctx: Send + Sync + 'static> ToolFunction<Ctx> for SkillsTool {
    fn name(&self) -> &'static str {
        SKILLS_TOOL_NAME
    }

    fn description(&self) -> &'static str {
        "Call this tool to load a skill, providing the name of the skill you are invoking"
    }

    fn input_schema(&self) -> serde_json::Value {
        skills_tool_input_schema()
    }

    async fn execute(
        &self,
        input: Value,
        _ctx: &Arc<ToolExecutionContext<Ctx>>,
    ) -> Result<ToolResult, AgentError> {
        let skill_name = input["skill_name"]
            .as_str()
            .ok_or_else(|| AgentError::ToolCallError("missing skill_name".into()))?;
        load_skill(skill_name, Path::new(SKILLS_PATH))
    }
}

#[async_trait::async_trait]
impl<Ctx: Send + Sync + 'static> ToolFunction<Ctx> for ConfiguredSkillsTool {
    fn name(&self) -> &'static str {
        SKILLS_TOOL_NAME
    }

    fn description(&self) -> &'static str {
        "Call this tool to load a skill, providing the name of the skill you are invoking"
    }

    fn input_schema(&self) -> serde_json::Value {
        skills_tool_input_schema()
    }

    async fn execute(
        &self,
        input: Value,
        _ctx: &Arc<ToolExecutionContext<Ctx>>,
    ) -> Result<ToolResult, AgentError> {
        let skill_name = input["skill_name"]
            .as_str()
            .ok_or_else(|| AgentError::ToolCallError("missing skill_name".into()))?;
        load_skill(skill_name, &self.skills_path)
    }
}

#[async_trait::async_trait]
impl<Ctx: Send + Sync + 'static> ToolFunction<Ctx> for TasksTool {
    fn name(&self) -> &'static str {
        TASKS_TOOL_NAME
    }

    fn description(&self) -> &'static str {
        "Tool for tracking and updating tasks throughout a session"
    }

    fn input_schema(&self) -> Value {
        json!({
          "type": "object",
          "required": [
            "task_name",
            "task_status"
          ],
          "properties": {
            "task_name": {
              "type": "string",
              "description": "Name of the task you are about to work on/currently working on/done with"
            },
            "task_status": {
              "type": "string",
              "enum": ["Queued", "InProgress", "Done"],
              "description": "Status of the task"
            }
          }
        })
    }

    async fn execute(
        &self,
        input: Value,
        _ctx: &Arc<ToolExecutionContext<Ctx>>,
    ) -> Result<ToolResult, AgentError> {
        let task_name = input["task_name"]
            .as_str()
            .ok_or_else(|| AgentError::ToolCallError("missing task_name".into()))?;
        let task_status = &input["task_status"];
        let ts: TaskStatus = serde_json::from_value(task_status.to_owned())
            .map_err(|_| AgentError::ToolCallError("invalid task status".into()))?;

        Ok(ToolResult::Ok(format!(
            "Successfully recorded task {} with status {}",
            task_name, ts
        )))
    }
}

#[async_trait::async_trait]
impl<Ctx: Send + Sync + 'static> Agent for MicroAgent<Ctx> {
    /// Generate the next assistant response as a raw token stream.
    ///
    /// The stream yields [`StreamChunk`]s that may contain text deltas or
    /// partial tool calls. Higher-level orchestration (e.g. [`run`]) is
    /// responsible for buffering and acting on tool calls.
    async fn generate(&mut self) -> Result<GenerationStream, AgentError> {
        let tools: Vec<Tool> = self
            .tools
            .values()
            .map(|t| t.to_sdk_tool())
            .collect::<Result<Vec<Tool>, AgentError>>()?;
        let mut request = LLMRequest::builder()
            .api_type(self.provider.clone().into())
            .api_key(self.api_key.clone())
            .model(self.model.clone())
            .stream(true)
            .messages(self.history.clone())
            .parallel_tool_calls(self.parallel_tool_calls)
            .build();
        request.base_url = self.base_url.clone();
        if !tools.is_empty() {
            request.tools = Some(tools);
        }
        if self.prompt_cache && self.provider == SupportedProvider::Anthropic {
            request.prompt_cache_ttl = Some("5m".to_string())
        }
        let stream = self
            .client
            .stream_response(request)
            .await
            .map_err(|e| AgentError::GenerationError(e.to_string()))?;
        let mapped =
            stream.map(|item| item.map_err(|e| AgentError::GenerationError(e.to_string())));
        Ok(Box::pin(mapped))
    }

    /// Run a complete conversation turn.
    ///
    /// If `session_id` is [`Some`] the conversation history is restored from
    /// storage; otherwise a new session is created. The returned stream yields
    /// high-level events ([`AgentEventAny`]) including deltas, tool calls,
    /// results, and the final stop event.
    async fn run(
        mut self,
        prompt: String,
        session_id: Option<String>,
    ) -> Result<RunStream, AgentError> {
        let local_tools: HashMap<String, Arc<dyn ToolFunction<Ctx>>> = self.tools.clone();
        let mut usage = Usage {
            input_tokens: 0,
            output_tokens: 0,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
            latency: 0,
        };
        let mut assistant_message: Option<Message> = None;
        let prompt = self.resolve_prompt(&prompt)?;
        let start_processing = Utc::now();
        let s: RunStream = Box::pin(stream! {
            let resolved_sid;
            let messages: Vec<Message> = if let Some(sid) = session_id {
                let ev = AgentEventAny::SessionInit(SessionInitEvent {
                    session_id: sid.clone(),
                    model: self.model.clone(),
                    system: self.system.clone(),
                    provider: self.provider.to_string(),
                    init_type: SessionInitType::Resume,
                    timestamp: Utc::now(),
                });
                yield Ok(ev);

                let events_res = self
                    .storage
                    .get_session(&sid)
                    .await
                    .map_err(|e| AgentError::SessionLoadError(e.to_string()));

                let events = match events_res {
                    Ok(e) => e,
                    Err(e) => {
                        yield Err(AgentError::RunError(format!("Error while getting the session: {}", e)));
                        return;
                    }
                };

                let incompl_tasks_res = get_incomplete_tasks(&events);
                let incomplete_tasks = match incompl_tasks_res {
                    Ok(t) => t,
                    Err(e) => {
                        yield Err(e);
                        return;
                    }
                };
                self.tasks = Arc::new(Mutex::new(incomplete_tasks));

                resolved_sid = sid;

                events
                    .iter()
                    .filter_map(|e| convert_event_to_message(e.clone()))
                    .collect()
            } else {
                let sid = uuid::Uuid::new_v4().to_string();
                let sint = SessionInitEvent {
                    session_id: sid.clone(),
                    model: self.model.clone(),
                    system: self.system.clone(),
                    provider: self.provider.to_string(),
                    init_type: SessionInitType::Start,
                    timestamp: Utc::now(),
                };
                resolved_sid = sid;
                let ev = AgentEventAny::SessionInit(sint.clone());
                match self.storage.create_session(sint).await {
                    Ok(_) => {},
                    Err(e) => {
                        yield Err(AgentError::RunError(format!("An error occurred while creating the session in the storage: {}", e)));
                        return;
                    }
                }
                yield Ok(ev);
                vec![]
            };
            self.history = messages;
            self.history.insert(0, Message { role: MessageRole::System, content: vec![MessagePart::Text(TextPart::new(self.system.clone()))] });
            self.history.push(Message {
                role: MessageRole::User,
                content: vec![MessagePart::Text(TextPart::new(prompt.clone()))],
            });
            let turn_id = uuid::Uuid::new_v4().to_string();
            let user_prompt_submit = AgentEventAny::UserPromptSubmit(UserPromptSubmitEvent {
                session_id: resolved_sid.clone(),
                turn_id: turn_id.clone(),
                prompt,
                timestamp: Utc::now(),
            });
                if let Err(error) = persist_event(self.storage.as_ref(), &user_prompt_submit).await {
                    yield Err(error);
                    return;
                }
            yield Ok(user_prompt_submit);

            loop {
                let mut generation = match self.generate().await {
                    Ok(g) => g,
                    Err(e) => {
                        yield Err(AgentError::RunError(format!("An error occurred while starting the generation stream: {}", e)));
                        return;
                    }
                };
                let mut tool_messages: Vec<Message> = vec![];
                let mut tool_calls: Vec<ToolCallPart> = vec![];
                while let Some(g) = generation.next().await {
                    match g {
                        Ok(chunk) => {
                            match chunk {
                                LLMStreamingResponse::Delta(d) => {
                                    if let Some(c) = d.delta {
                                        let ev = AgentEventAny::StreamDelta(StreamDeltaEvent {
                                            session_id: resolved_sid.clone(),
                                            turn_id: turn_id.clone(),
                                            delta: c,
                                            delta_type: DeltaType::Text,
                                            timestamp: Utc::now(),
                                        });
                                        if let Err(error) = persist_event(self.storage.as_ref(), &ev).await {
                                            yield Err(error);
                                            return;
                                        }
                                        yield Ok(ev);
                                    }
                                }
                                LLMStreamingResponse::ToolDelta(_) => {}
                                LLMStreamingResponse::ThinkingDelta(d) => {
                                    if let Some(c) = d.delta {
                                        let ev = AgentEventAny::StreamDelta(StreamDeltaEvent {
                                            session_id: resolved_sid.clone(),
                                            turn_id: turn_id.clone(),
                                            delta: c,
                                            delta_type: DeltaType::Thinking,
                                            timestamp: Utc::now(),
                                        });
                                        if let Err(error) = persist_event(self.storage.as_ref(), &ev).await {
                                            yield Err(error);
                                            return;
                                        }
                                        yield Ok(ev);
                                    }
                                }
                                LLMStreamingResponse::Complete(c) => {
                                    if let Some(u) = c.usage {
                                        usage.input_tokens += u.input_tokens;
                                        usage.output_tokens += u.output_tokens;
                                        usage.cache_read_tokens += u.cache_read_tokens.unwrap_or_default();
                                        usage.cache_write_tokens += u.cache_write_tokens.unwrap_or_default();
                                    }
                                    assistant_message = Some(c.message);
                                    tool_calls = c.tool_calls.unwrap_or(vec![]);
                                }
                            }
                        },
                        Err(e) => {
                            usage.latency = (Utc::now() - start_processing).num_milliseconds();
                            let stop_ev = AgentEventAny::SessionStop(SessionStopEvent { session_id: resolved_sid.clone(), success: false, result: None, error: Some(e.to_string()), timestamp: Utc::now(), usage, incomplete_tasks: {
                                let ts = self.tasks.lock().await;
                                if ts.is_empty() {
                                    None
                                } else {
                                    Some(ts.keys().map(|k| k.to_string()).collect())
                                }
                            }});
                            if let Err(error) = persist_event(self.storage.as_ref(), &stop_ev).await {
                                yield Err(error);
                                return;
                            }
                            yield Ok(stop_ev);
                            return;
                        }
                    }
                }

                let Some(final_message) = assistant_message.take() else {
                    yield Err(AgentError::RunError("LLM stream did not produce a full message".to_string()));
                    return;
                };

                if tool_calls.is_empty() {
                    usage.latency = (Utc::now() - start_processing).num_milliseconds();
                    let content = match assistant_message_parts(&final_message) {
                        Ok(content) => content,
                        Err(error) => {
                            yield Err(error);
                            return;
                        }
                    };
                    let ev = AgentEventAny::AssistantResponse(AssistantResponseEvent {
                        session_id: resolved_sid.clone(),
                        turn_id: turn_id.clone(),
                        content: content.clone(),
                        timestamp: Utc::now(),
                    });
                    let stop_ev = AgentEventAny::SessionStop(SessionStopEvent {
                        session_id: resolved_sid.clone(),
                        success: true,
                        result: Some(content),
                        error: None,
                        timestamp: Utc::now(),
                        usage,
                        incomplete_tasks: {
                            let ts = self.tasks.lock().await;
                            if ts.is_empty() {
                                None
                            } else {
                                Some(ts.keys().map(|k| k.to_string()).collect())
                            }
                        }
                    });
                    if let Err(error) = persist_event(self.storage.as_ref(), &ev).await {
                        yield Err(error);
                        return;
                    }
                    if let Err(error) = persist_event(self.storage.as_ref(), &stop_ev).await {
                        yield Err(error);
                        return;
                    }
                    yield Ok(ev);
                    yield Ok(stop_ev);
                    return;
                }

                let mut to_call = JoinSet::new();
                let tool_ctx = self.tool_context.clone();
                let concurrency = if !self.parallel_tool_calls {
                    1
                } else {
                    MAX_CONCURRENT_TOOL_CALLS
                };
                let semaphore = Arc::new(Semaphore::new(concurrency));
                for tc in tool_calls {
                            let v: Value = match serde_json::from_str(&tc.arguments) {
                                Ok(value) => value,
                                Err(error) => {
                                    yield Err(AgentError::RunError(format!(
                                        "Tool {} returned invalid JSON arguments: {error}",
                                        tc.name
                                    )));
                                    return;
                                }
                            };
                            let tool = local_tools.get(&tc.name);
                            if let Some(t) = tool {
                                let tool_name = tc.name.clone();
                                let tc_any_ev = AgentEventAny::ToolAnyCall(ToolCallAnyEvent {
                                    session_id: resolved_sid.clone(),
                                    turn_id: turn_id.clone(),
                                    name: tool_name.clone(),
                                    input: v.clone(),
                                    timestamp: Utc::now(),
                                    tool_call_id: tc.id.clone(),
                                });
                                if let Err(error) = persist_event(self.storage.as_ref(), &tc_any_ev).await {
                                    yield Err(error);
                                    return;
                                }
                                let tc_ev = if tool_name != SKILLS_TOOL_NAME && tool_name != TASKS_TOOL_NAME {
                                    AgentEventAny::ToolCall(ToolCallEvent {
                                        session_id: resolved_sid.clone(),
                                        turn_id: turn_id.clone(),
                                        name: tool_name,
                                        input: v.clone(),
                                        timestamp: Utc::now(),
                                        tool_call_id: tc.id.clone(),
                                    })
                                } else if tool_name == SKILLS_TOOL_NAME {
                                    match v["skill_name"].as_str() {
                                        Some(n) => AgentEventAny::SkillLoad(SkillLoadEvent {
                                            session_id: resolved_sid.clone(),
                                            turn_id: turn_id.clone(),
                                            skill_name: n.to_string(),
                                            timestamp: Utc::now(),
                                        }),
                                        None => {
                                            yield Err(AgentError::RunError("Skill name is not a string".to_string()));
                                            return;
                                        }
                                    }
                                } else {
                                    let tn = match v["task_name"].as_str() {
                                        Some(n) => n,
                                        None => {
                                            yield Err(AgentError::RunError("Task name is not a string".to_string()));
                                            return;
                                        }
                                    };
                                    let ts: TaskStatus = match serde_json::from_value(v["task_status"].to_owned()) {
                                        Ok(v) => v,
                                        Err(_) => {
                                            yield Err(AgentError::RunError("Invalid type for task status".to_string()));
                                            return;
                                        }
                                    };
                                    let mut tasks = self.tasks.lock().await;
                                    if ts == TaskStatus::Done {
                                        tasks.remove(tn);
                                    } else {
                                        tasks.entry(tn.to_string()).and_modify(|v| *v = ts).or_insert(ts);
                                    }
                                    AgentEventAny::Task(TaskEvent {
                                        session_id: resolved_sid.clone(),
                                        turn_id: turn_id.clone(),
                                        task_name: tn.to_string(),
                                        task_status: ts,
                                        timestamp: Utc::now(),
                                    })
                                };
                                if let Err(error) = persist_event(self.storage.as_ref(), &tc_ev).await {
                                    yield Err(error);
                                    return;
                                }
                                yield Ok(tc_ev);
                                let permit_res = semaphore.clone().acquire_owned().await;
                                let permit = match permit_res {
                                    Ok(p) => p,
                                    Err(e) => {
                                        yield Err(AgentError::RunError(format!("Error while acquiring semaphore: {}", e)));
                                        return;
                                    }
                                };
                                let t = t.clone();
                                let tool_call_id = tc.id.clone();
                                let ctx = tool_ctx.clone();
                                to_call.spawn(async move {
                                    let _permit = permit;
                                    let result = call_tool(t, v, ctx).await;
                                    match result {
                                        Ok(r) => Ok((tool_call_id, r)),
                                        Err(e) => Err(e)
                                    }
                                });
                            }
                    }
                while let Some(res) = to_call.join_next().await {
                    match res {
                        Ok(Ok((tid, tool_result))) => {
                            let ev = AgentEventAny::ToolResult(ToolResultEvent {
                                session_id: resolved_sid.clone(),
                                turn_id: turn_id.clone(),
                                result: tool_result.clone(),
                                tool_call_id: tid.clone(),
                                timestamp: Utc::now(),
                            });
                            if let Err(error) = persist_event(self.storage.as_ref(), &ev).await {
                                yield Err(error);
                                return;
                            }
                            yield Ok(ev);
                            let content = match tool_result {
                                ToolResult::Ok(r) => {
                                    format!("Tool succeeded: {r}")
                                },
                                ToolResult::Err(r) => {
                                    format!("Tool failed: {r}")
                                },
                                _ => {
                                    yield Err(AgentError::RunError(
                                        "Tool returned an unsupported result".to_string(),
                                    ));
                                    return;
                                }
                            };
                            tool_messages.push(Message { role: MessageRole::Tool, content: vec![MessagePart::ToolResult(ToolResultPart {
                                tool_call_id: tid,
                                result: content,
                            })] });
                        }
                        Ok(Err(e)) => {
                            yield Err(AgentError::RunError(format!("Tool call failed: {}", e)));
                        }
                        Err(e) => {
                            yield Err(AgentError::RunError(format!("Task join failed: {}", e)));
                        }
                    }
                }

                self.history.push(final_message);
                self.history.extend(tool_messages);
            }
        });
        Ok(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        Agent, AgentError, GenerationStream, RunStream, ToolExecutionContext, ToolFunction,
    };
    use async_stream::stream;
    use futures_util::StreamExt;
    use llms_sdk::LLMStreamingDelta;
    use microagents_events::types::ToolResult;
    use serde_json::Value;
    use std::sync::Arc;

    // ------------------------------------------------------------------
    // DummyAgent – a mock implementation of the Agent trait
    // ------------------------------------------------------------------

    #[derive(Debug)]
    struct DummyAgent {
        pub generate_called: bool,
        pub run_called: bool,
        pub last_prompt: Option<String>,
        pub last_session_id: Option<String>,
    }

    impl DummyAgent {
        fn new() -> Self {
            Self {
                generate_called: false,
                run_called: false,
                last_prompt: None,
                last_session_id: None,
            }
        }
    }

    #[async_trait::async_trait]
    impl Agent for DummyAgent {
        async fn generate(&mut self) -> Result<GenerationStream, AgentError> {
            self.generate_called = true;
            let s = stream! {
                yield Ok(LLMStreamingResponse::Delta(LLMStreamingDelta {
                    stop: false,
                    response_id: "1".to_string(),
                    created_at: None,
                    delta: Some("something".to_string())
                }));
            };
            Ok(Box::pin(s))
        }

        async fn run(
            mut self,
            prompt: String,
            session_id: Option<String>,
        ) -> Result<RunStream, AgentError> {
            self.run_called = true;
            self.last_prompt = Some(prompt.clone());
            self.last_session_id = session_id.clone();
            let s = stream! {
                yield Ok(AgentEventAny::UserPromptSubmit(UserPromptSubmitEvent {
                    session_id: session_id.unwrap_or_else(|| "new".into()),
                    turn_id: "t1".into(),
                    prompt,
                    timestamp: Utc::now(),
                }));
            };
            Ok(Box::pin(s))
        }
    }

    // ------------------------------------------------------------------
    // A simple dummy tool for builder tests
    // ------------------------------------------------------------------

    #[derive(Debug)]
    struct DummyTool;

    #[async_trait::async_trait]
    impl ToolFunction<()> for DummyTool {
        fn name(&self) -> &'static str {
            "dummy"
        }
        fn description(&self) -> &'static str {
            "A dummy tool"
        }
        fn input_schema(&self) -> Value {
            json!({"type": "object"})
        }
        async fn execute(
            &self,
            _input: Value,
            _ctx: &Arc<ToolExecutionContext<()>>,
        ) -> Result<ToolResult, AgentError> {
            Ok(ToolResult::Ok("done".into()))
        }
    }

    // ------------------------------------------------------------------
    // Builder default tests
    // ------------------------------------------------------------------

    #[test]
    fn test_builder_default_provider_is_openai() {
        let builder = MicroAgentBuilder::new(ToolExecutionContext::new(()));
        assert_eq!(builder.provider, SupportedProvider::OpenAI);
    }

    #[test]
    fn test_builder_default_model_is_empty() {
        let builder = MicroAgentBuilder::new(ToolExecutionContext::new(()));
        assert!(builder.model.is_empty());
    }

    #[test]
    fn test_builder_default_skills_is_empty() {
        let builder = MicroAgentBuilder::new(ToolExecutionContext::new(()));
        assert!(builder.skills.is_empty());
    }

    #[test]
    fn test_builder_default_tools_contains_skills_and_tasks_tool() {
        let builder = MicroAgentBuilder::new(ToolExecutionContext::new(()));
        assert!(builder.tools.contains_key("skills"));
        assert!(builder.tools.contains_key("tasks"));
        assert_eq!(builder.tools.len(), 2);
    }

    #[test]
    fn test_builder_default_parallel_tool_calls_is_false() {
        let builder = MicroAgentBuilder::new(ToolExecutionContext::new(()));
        assert!(!builder.parallel_tool_calls);
    }

    // ------------------------------------------------------------------
    // Builder pattern tests
    // ------------------------------------------------------------------

    #[test]
    fn test_builder_provider_invalid_returns_error() {
        let result =
            MicroAgentBuilder::new(ToolExecutionContext::new(())).provider("unknown".to_string());
        assert!(result.is_err());
        assert!(
            matches!(
                result.unwrap_err(),
                MicroAgentBuilderError::ProviderNotSupported(_)
            ),
            "expected ProviderNotSupported error"
        );
    }

    #[test]
    fn test_builder_with_supported_provider_as_str() {
        let builder = MicroAgentBuilder::new(ToolExecutionContext::new(()))
            .provider("anthropic".to_string())
            .unwrap();
        assert_eq!(builder.provider, SupportedProvider::Anthropic);
    }

    #[test]
    fn test_builder_with_supported_provider_as_enum() {
        let builder = MicroAgentBuilder::new(ToolExecutionContext::new(()))
            .provider(SupportedProvider::OpenAI)
            .unwrap();
        assert_eq!(builder.provider, SupportedProvider::OpenAI);
    }

    #[test]
    fn test_builder_model_sets_model() {
        let builder = MicroAgentBuilder::new(ToolExecutionContext::new(())).model("gpt-5.5");
        assert_eq!(builder.model, "gpt-5.5");
    }

    #[test]
    fn test_builder_parallel_tool_calls_sets_flag() {
        let builder =
            MicroAgentBuilder::new(ToolExecutionContext::new(())).parallel_tool_calls(true);
        assert!(builder.parallel_tool_calls);
    }

    #[test]
    fn test_builder_prompt_cache_sets_flag() {
        let builder = MicroAgentBuilder::new(ToolExecutionContext::new(())).prompt_cache(true);
        assert!(builder.prompt_cache);
    }

    #[test]
    fn test_builder_base_url_gets_set() {
        let builder = MicroAgentBuilder::new(ToolExecutionContext::new(()))
            .base_url("https://api.anthropic.com/v1");
        assert_eq!(
            builder.base_url,
            Some("https://api.anthropic.com/v1".to_string())
        );
    }

    #[test]
    fn test_builder_custom_instructions_sets_instructions() {
        let builder =
            MicroAgentBuilder::new(ToolExecutionContext::new(())).custom_instructions("Be concise");
        assert_eq!(builder.custom_instructions, "Be concise");
    }

    #[test]
    fn test_builder_add_tool_increments_tools() {
        let builder = MicroAgentBuilder::new(ToolExecutionContext::new(()))
            .add_tool(Arc::new(DummyTool))
            .unwrap();
        assert_eq!(builder.tools.len(), 3);
        assert!(builder.tools.contains_key("dummy"));
    }

    #[test]
    fn test_builder_add_tool_rejects_duplicate_name() {
        let result = MicroAgentBuilder::new(ToolExecutionContext::new(()))
            .add_tool(Arc::new(DummyTool))
            .and_then(|builder| builder.add_tool(Arc::new(DummyTool)));

        assert!(matches!(
            result,
            Err(MicroAgentBuilderError::ToolAlreadyDefined(name)) if name == "dummy"
        ));
    }

    #[test]
    fn test_builder_add_skill_uses_configured_skills_path() {
        let temp = tempfile::tempdir().expect("temporary directory should be created");
        let skills_path = temp.path().join("project-skills");
        let skill_path = skills_path.join("configured-skill");
        std::fs::create_dir_all(&skill_path).expect("skill directory should be created");
        std::fs::write(
            skill_path.join("SKILL.md"),
            "---\nname: configured-skill\ndescription: configured\n---\n",
        )
        .expect("skill file should be written");

        let builder = MicroAgentBuilder::new(ToolExecutionContext::new(()))
            .skills_path(&skills_path)
            .add_skill("configured-skill")
            .expect("skill should be loaded from the configured path");

        assert_eq!(
            builder.skills.get("configured-skill"),
            Some(&"configured".to_string())
        );
    }

    #[test]
    fn test_builder_add_skill_rejects_invalid_name() {
        let result = MicroAgentBuilder::new(ToolExecutionContext::new(())).add_skill("../outside");

        assert!(matches!(
            result,
            Err(MicroAgentBuilderError::InvalidSkillName(name)) if name == "../outside"
        ));
    }

    #[tokio::test]
    async fn test_configured_skills_tool_uses_configured_skills_path() {
        let temp = tempfile::tempdir().expect("temporary directory should be created");
        let skills_path = temp.path().join("project-skills");
        let skill_path = skills_path.join("configured-skill");
        std::fs::create_dir_all(&skill_path).expect("skill directory should be created");
        std::fs::write(skill_path.join("SKILL.md"), "configured skill content")
            .expect("skill file should be written");

        let builder =
            MicroAgentBuilder::new(ToolExecutionContext::new(())).skills_path(&skills_path);
        let skills_tool = builder
            .tools
            .get(SKILLS_TOOL_NAME)
            .expect("skills tool should be registered");
        let result = skills_tool
            .execute(
                json!({"skill_name": "configured-skill"}),
                &builder.tool_context,
            )
            .await
            .expect("skills tool should load from the configured path");

        assert!(matches!(result, ToolResult::Ok(content) if content == "configured skill content"));
    }

    #[tokio::test]
    async fn test_configured_skills_tool_rejects_path_traversal() {
        let temp = tempfile::tempdir().expect("temporary directory should be created");
        let skills_path = temp.path().join("project-skills");
        let outside_skill = temp.path().join("outside-skill");
        std::fs::create_dir_all(&skills_path).expect("project skills directory should be created");
        std::fs::create_dir_all(&outside_skill).expect("outside skill directory should be created");
        std::fs::write(outside_skill.join("SKILL.md"), "outside skill content")
            .expect("outside skill file should be written");

        let builder =
            MicroAgentBuilder::new(ToolExecutionContext::new(())).skills_path(&skills_path);
        let skills_tool = builder
            .tools
            .get(SKILLS_TOOL_NAME)
            .expect("skills tool should be registered");
        let result = skills_tool
            .execute(
                json!({"skill_name": "../outside-skill"}),
                &builder.tool_context,
            )
            .await
            .expect("invalid skill names are reported as tool results");

        assert!(matches!(result, ToolResult::Err(message) if message.contains("invalid")));
    }

    #[tokio::test]
    async fn test_persist_event_returns_run_error_when_storage_update_fails() {
        let storage = InMemoryAgentStorage::default();
        let event = AgentEventAny::UserPromptSubmit(UserPromptSubmitEvent {
            session_id: "missing-session".to_string(),
            turn_id: "turn-1".to_string(),
            prompt: "hello".to_string(),
            timestamp: Utc::now(),
        });

        let error = persist_event(&storage, &event)
            .await
            .expect_err("updating an unknown session must fail");

        assert!(
            matches!(error, AgentError::RunError(message) if message.contains("updating the session"))
        );
    }

    #[test]
    fn test_assistant_message_parts_converts_text() {
        let message = Message {
            role: MessageRole::Assistant,
            content: vec![MessagePart::Text(TextPart::new("hello"))],
        };

        let parts = assistant_message_parts(&message).expect("text is a supported assistant part");

        assert_eq!(
            parts,
            vec![AssistantMessagePart::Text(AssistantTextPart {
                text: "hello".to_string(),
            })]
        );
    }

    #[tokio::test]
    async fn test_builder_storage_sets_jsonl() {
        let builder = MicroAgentBuilder::new(ToolExecutionContext::new(()))
            .storage(AgentStorageChoice::Jsonl)
            .await
            .unwrap();
        // We cannot directly inspect the dyn type, but building should succeed
        let _agent = builder.build().expect("Should be able to build the agent");
    }

    #[tokio::test]
    async fn test_builder_storage_sets_memory() {
        let builder = MicroAgentBuilder::new(ToolExecutionContext::new(()))
            .storage(AgentStorageChoice::Memory)
            .await
            .unwrap();
        let _agent = builder.build().expect("Should be able to build the agent");
    }

    #[tokio::test]
    async fn test_builder_storage_sets_sqlite() {
        let builder = MicroAgentBuilder::new(ToolExecutionContext::new(()))
            .storage(AgentStorageChoice::Sqlite)
            .await
            .unwrap();
        let _agent = builder.build().expect("Should be able to build the agent");
    }

    // ------------------------------------------------------------------
    // Build / resolve tests
    // ------------------------------------------------------------------

    #[test]
    fn test_build_sets_empty_history() {
        let original_value = std::env::var("OPENAI_API_KEY").unwrap_or_default();
        unsafe {
            std::env::set_var("OPENAI_API_KEY", "test");
        }
        let agent = MicroAgentBuilder::new(ToolExecutionContext::new(()))
            .provider("openai".to_string())
            .unwrap()
            .build()
            .expect("Should be able to build the agent");
        assert!(agent.history.is_empty());
        unsafe {
            std::env::set_var("OPENAI_API_KEY", original_value);
        }
    }

    #[test]
    fn test_build_sets_tools_on_agent() {
        let original_value = std::env::var("OPENAI_API_KEY").unwrap_or_default();
        unsafe {
            std::env::set_var("OPENAI_API_KEY", "test");
        }
        let agent = MicroAgentBuilder::new(ToolExecutionContext::new(()))
            .provider("openai".to_string())
            .unwrap()
            .add_tool(Arc::new(DummyTool))
            .unwrap()
            .build()
            .expect("Should be able to build the agent");
        assert_eq!(agent.tools.len(), 3);
        unsafe {
            std::env::set_var("OPENAI_API_KEY", original_value);
        }
    }

    #[test]
    fn test_build_sets_provider_on_agent() {
        let original_value = std::env::var("OPENAI_API_KEY").unwrap_or_default();
        unsafe {
            std::env::set_var("OPENAI_API_KEY", "test");
        }
        let agent = MicroAgentBuilder::new(ToolExecutionContext::new(()))
            .provider("openai".to_string())
            .unwrap()
            .build()
            .expect("Should be able to build the agent");
        assert_eq!(agent.provider, SupportedProvider::OpenAI);
        unsafe {
            std::env::set_var("OPENAI_API_KEY", original_value);
        }
    }

    #[test]
    fn test_build_sets_model_on_agent() {
        let original_value = std::env::var("OPENAI_API_KEY").unwrap_or_default();
        unsafe {
            std::env::set_var("OPENAI_API_KEY", "test");
        }
        let agent = MicroAgentBuilder::new(ToolExecutionContext::new(()))
            .provider("openai".to_string())
            .unwrap()
            .model("gpt-4.1")
            .build()
            .expect("Should be able to build the agent");
        assert_eq!(agent.model, "gpt-4.1");
        unsafe {
            std::env::set_var("OPENAI_API_KEY", original_value);
        }
    }

    #[test]
    fn test_build_sets_parallel_tool_calls_on_agent() {
        let original_value = std::env::var("OPENAI_API_KEY").unwrap_or_default();
        unsafe {
            std::env::set_var("OPENAI_API_KEY", "test");
        }
        let agent = MicroAgentBuilder::new(ToolExecutionContext::new(()))
            .provider("openai".to_string())
            .unwrap()
            .parallel_tool_calls(true)
            .build()
            .expect("Should be able to build the agent");
        assert!(agent.parallel_tool_calls);
        unsafe {
            std::env::set_var("OPENAI_API_KEY", original_value);
        }
    }

    #[test]
    fn test_build_system_prompt_contains_base() {
        let original_value = std::env::var("OPENAI_API_KEY").unwrap_or_default();
        unsafe {
            std::env::set_var("OPENAI_API_KEY", "test");
        }
        let agent = MicroAgentBuilder::new(ToolExecutionContext::new(()))
            .provider("openai".to_string())
            .unwrap()
            .build()
            .expect("Should be able to build the agent");
        assert!(agent.system.contains("You are MicroAgent"));
        unsafe {
            std::env::set_var("OPENAI_API_KEY", original_value);
        }
    }

    #[test]
    fn test_build_system_prompt_contains_tools() {
        let original_value = std::env::var("OPENAI_API_KEY").unwrap_or_default();
        unsafe {
            std::env::set_var("OPENAI_API_KEY", "test");
        }
        let agent = MicroAgentBuilder::new(ToolExecutionContext::new(()))
            .provider("openai".to_string())
            .unwrap()
            .add_tool(Arc::new(DummyTool))
            .unwrap()
            .build()
            .expect("Should be able to build the agent");
        assert!(agent.system.contains("<tools>"));
        assert!(agent.system.contains("<name>dummy</name>"));
        unsafe {
            std::env::set_var("OPENAI_API_KEY", original_value);
        }
    }

    #[test]
    fn test_build_system_prompt_contains_default_model_when_model_empty() {
        let original_value = std::env::var("OPENAI_API_KEY").unwrap_or_default();
        unsafe {
            std::env::set_var("OPENAI_API_KEY", "test");
        }
        let agent = MicroAgentBuilder::new(ToolExecutionContext::new(()))
            .build()
            .expect("Should be able to build the agent");
        // Default provider is OpenAI -> default model is gpt-5.6-terra
        assert!(agent.system.contains("gpt-5.6-terra"));
        unsafe {
            std::env::set_var("OPENAI_API_KEY", original_value);
        }
    }

    #[test]
    fn test_build_system_prompt_contains_custom_model_when_set() {
        let original_value = std::env::var("OPENAI_API_KEY").unwrap_or_default();
        unsafe {
            std::env::set_var("OPENAI_API_KEY", "test");
        }
        let agent = MicroAgentBuilder::new(ToolExecutionContext::new(()))
            .provider("openai".to_string())
            .unwrap()
            .model("custom-model")
            .build()
            .expect("Should be able to build the agent");
        assert!(agent.system.contains("custom-model"));
        assert!(!agent.system.contains("gpt-5.6-terra"));
        unsafe {
            std::env::set_var("OPENAI_API_KEY", original_value);
        }
    }

    #[test]
    fn test_agent_fails_to_build_if_not_api_key() {
        let result = MicroAgentBuilder::new(ToolExecutionContext::new(()))
            .provider("anthropic".to_string())
            .unwrap()
            .build();
        assert!(result.is_err_and(|e| matches!(e, MicroAgentBuilderError::EnvVarNotFoundError(_))));
    }

    // ------------------------------------------------------------------
    // SupportedProvider tests
    // ------------------------------------------------------------------

    #[test]
    fn test_supported_provider_from_str_valid() {
        assert_eq!(
            SupportedProvider::from_str("openai").unwrap(),
            SupportedProvider::OpenAI
        );
        assert_eq!(
            SupportedProvider::from_str("anthropic").unwrap(),
            SupportedProvider::Anthropic
        );
    }

    #[test]
    fn test_supported_provider_from_str_invalid() {
        assert!(SupportedProvider::from_str("azure").is_err());
    }

    #[test]
    fn test_supported_provider_display() {
        assert_eq!(SupportedProvider::OpenAI.to_string(), "openai");
        assert_eq!(SupportedProvider::Anthropic.to_string(), "anthropic");
    }

    #[test]
    fn test_supported_provider_default_model() {
        assert_eq!(
            SupportedProvider::OpenAI.default_model().unwrap(),
            "gpt-5.6-terra"
        );
        assert_eq!(
            SupportedProvider::Anthropic.default_model().unwrap(),
            "claude-sonnet-5"
        );
    }

    #[test]
    fn test_supported_provider_default_is_openai() {
        let provider: SupportedProvider = Default::default();
        assert_eq!(provider, SupportedProvider::OpenAI);
    }

    // ------------------------------------------------------------------
    // DummyAgent mock tests
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn test_dummy_agent_generate_sets_flag() {
        let mut agent = DummyAgent::new();
        assert!(!agent.generate_called);
        let _ = agent.generate().await;
        assert!(agent.generate_called);
    }

    #[tokio::test]
    async fn test_dummy_agent_generate_returns_stream() {
        let mut agent = DummyAgent::new();
        let mut stream = agent.generate().await.unwrap();
        let item = stream.next().await;
        assert!(item.is_some());
    }

    #[tokio::test]
    async fn test_dummy_agent_run_streams_prompt() {
        let agent = DummyAgent::new();
        let mut stream = agent
            .run("hello".into(), Some("sid-123".into()))
            .await
            .unwrap();
        // We consumed self in run, so we can't check the fields directly.
        // Instead we verify via the yielded event.
        let item = stream.next().await.unwrap().unwrap();
        match item {
            AgentEventAny::UserPromptSubmit(ev) => {
                assert_eq!(ev.prompt, "hello");
                assert_eq!(ev.session_id, "sid-123");
            }
            _ => panic!("expected UserPromptSubmit"),
        }
    }

    #[tokio::test]
    async fn test_dummy_agent_run_with_none_session_id() {
        let agent = DummyAgent::new();
        let mut stream = agent.run("test".into(), None).await.unwrap();
        let item = stream.next().await.unwrap().unwrap();
        match item {
            AgentEventAny::UserPromptSubmit(ev) => {
                assert_eq!(ev.session_id, "new");
            }
            _ => panic!("expected UserPromptSubmit"),
        }
    }

    #[tokio::test]
    async fn test_dummy_agent_run_stream_yields_single_event() {
        let agent = DummyAgent::new();
        let mut stream = agent.run("prompt".into(), None).await.unwrap();
        let first = stream.next().await;
        assert!(first.is_some());
        let second = stream.next().await;
        assert!(second.is_none());
    }
}
