use llms_sdk::{LLMStreamingResponse, Tool as SdkTool};
use microagents_events::{AgentEventAny, types::ToolResult};
use serde_json::Value;
use std::{env::VarError, fmt::Debug, pin::Pin, sync::Arc};
use thiserror::Error;

use crate::agent::MicroAgentBuilderError;

/// Errors that can occur during agent execution.
#[derive(Error, Debug)]
pub enum AgentError {
    /// The LLM generation step failed.
    #[error("Generation failed because of {0}")]
    GenerationError(String),
    /// A skill could not be resolved.
    #[error("Skill resolution failed")]
    SkillResolutionError,
    /// A tool call failed.
    #[error("Tool call failed because of {0}")]
    ToolCallError(String),
    /// The overall run failed.
    #[error("Run failure: {0}")]
    RunError(String),
    /// The LLM client could not be initialized.
    #[error("Unable to initialize client: {0}")]
    ClientInitFailed(String),
    /// MicroAgentBuilder error
    #[error(transparent)]
    BuilderError(#[from] MicroAgentBuilderError),
    /// An API key environment variable was missing.
    #[error(transparent)]
    ApiKeyNotConfigured(#[from] VarError),
    /// Loading a persisted session failed.
    #[error("Session load error: {0}")]
    SessionLoadError(String),
    /// Converting an event to a chat message failed.
    #[error("Conversion error from agent event to message")]
    EventConversionError,
    /// Tokenizer could not be loaded
    #[error("Tokenizer could not be loaded: {0}")]
    TokenizerLoadingError(String),
    #[error("Error loading tasks: {0}")]
    TaskLoadingError(String),
    #[error("Error while converting tool to sdk tool: {0}")]
    ToolConversionError(String),
}

/// A single item yielded by a [`GenerationStream`].
pub type StreamItem = Result<LLMStreamingResponse, AgentError>;
/// Stream of raw LLM chunks returned by [`Agent::generate`].
pub type GenerationStream = Pin<Box<dyn futures_core::Stream<Item = StreamItem> + Send>>;
/// A single event yielded by a [`RunStream`].
pub type RunStreamItem = Result<AgentEventAny, AgentError>;
/// Stream of high-level agent events returned by [`Agent::run`].
pub type RunStream = Pin<Box<dyn futures_core::Stream<Item = RunStreamItem> + Send>>;

/// Core trait for agent implementations.
///
/// [`generate`] performs a single LLM call and returns the raw token stream.
/// [`run`] executes a full turn (or multi-turn) conversation, handling tool
/// calls, skills, and session persistence automatically.
#[async_trait::async_trait]
pub trait Agent: Send + Sync {
    /// Generate the next assistant response as a raw token stream.
    async fn generate(&mut self) -> Result<GenerationStream, AgentError>;
    /// Run a complete conversation turn, optionally resuming an existing session.
    async fn run(
        mut self,
        prompt: String,
        session_id: Option<String>,
    ) -> Result<RunStream, AgentError>;
}

/// Context passed to every [`ToolFunction`] invocation.
///
/// `Ctx` is user-defined state (e.g. a database handle, HTTP client, etc.).
#[derive(Debug, Clone)]
pub struct ToolExecutionContext<Ctx> {
    pub context: Ctx,
}

impl<Ctx> ToolExecutionContext<Ctx> {
    /// Wrap user context so it can be shared with tools.
    pub fn new(context: Ctx) -> Self {
        Self { context }
    }
}

impl<Ctx: Default> Default for ToolExecutionContext<Ctx> {
    fn default() -> Self {
        Self::new(Ctx::default())
    }
}

/// A callable tool that the agent can invoke.
///
/// Implementors define the JSON schema, description, and execution logic.
/// The agent automatically serialises the LLM's arguments and validates them
/// against [`input_schema`] before calling [`execute`].
#[async_trait::async_trait]
pub trait ToolFunction<Ctx>: Debug + Send + Sync {
    /// Unique tool name exposed to the LLM.
    fn name(&self) -> &'static str;
    /// Human-readable description exposed to the LLM.
    fn description(&self) -> &'static str;
    /// JSON Schema describing the arguments this tool accepts.
    fn input_schema(&self) -> Value;
    /// Run the tool with the validated arguments.
    async fn execute(
        &self,
        input: Value,
        ctx: &Arc<ToolExecutionContext<Ctx>>,
    ) -> Result<ToolResult, AgentError>;
    /// Convert this tool into the SDK representation used for chat requests.
    fn to_sdk_tool(&self) -> Result<SdkTool, AgentError> {
        Ok(SdkTool {
            name: self.name().to_string(),
            description: self.description().to_string(),
            parameters: self
                .input_schema()
                .try_into()
                .map_err(|e: serde_json::Error| AgentError::ToolConversionError(e.to_string()))?,
        })
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[derive(Debug)]
    struct MyTool;

    #[async_trait::async_trait]
    impl ToolFunction<()> for MyTool {
        fn name(&self) -> &'static str {
            "tool"
        }

        fn description(&self) -> &'static str {
            "description"
        }

        fn input_schema(&self) -> Value {
            json!({
              "$schema": "https://json-schema.org/draft-07/schema",
              "type": "object",
              "properties": {
                "hobbies": {
                  "type": "array",
                  "description": "List of hobbies",
                  "items": {
                    "type": "object",
                    "properties": {
                      "name": {
                        "type": "string",
                        "description": "Name of the hobby"
                      },
                      "yearsExperience": {
                        "type": "number",
                        "description": "Years of experience"
                      }
                    }
                  }
                }
              },
              "required": []
            })
        }

        async fn execute(
            &self,
            _input: Value,
            _ctx: &Arc<ToolExecutionContext<()>>,
        ) -> Result<ToolResult, AgentError> {
            Ok(ToolResult::Ok("success".into()))
        }
    }

    #[test]
    fn test_to_sdk_tool() {
        let tool = MyTool {};
        let sdk_tool = tool
            .to_sdk_tool()
            .expect("Should be able to convert to tool");
        assert_eq!(sdk_tool.name, tool.name().to_owned());
        assert_eq!(sdk_tool.description, tool.description().to_owned());
        assert_eq!(sdk_tool.parameters, tool.input_schema());
    }
}
