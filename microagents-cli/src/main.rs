use clap::Parser;
use futures_util::StreamExt;
use microagents_core::{
    agent::{MicroAgentBuilder, SupportedProvider},
    types::{Agent, AgentError, ToolExecutionContext},
};
use microagents_events::types::AgentEvent;
use microagents_storage::types::AgentStorageChoice;
use std::{str::FromStr, sync::Arc};

use crate::init_env::{infer_provider_from_env, initialize_environment};

mod init_env;
mod processing;
mod search;
mod tools;
mod tui;

/// CLI Agent built on top of the MicroAgents framework
#[derive(Parser, Debug)]
#[command(version = "0.1.0")]
#[command(name = "microag")]
#[command(about, long_about = None)]
struct Args {
    /// LLM model to use (must match with the one offered by the provider).
    /// Falls back to the provider's default model if not provided
    #[arg(long, default_value = None)]
    model: Option<String>,

    /// Add one or more local/global skills by providing their name.
    /// Falls back to skills auto-discovery if no skill has been provided.
    #[arg(long)]
    skill: Vec<String>,

    /// Provider to use for the agent, either 'openai' or 'anthropic'.
    /// If the provider is omitted, tries to load it from the environment and,
    /// if not found in the environment, falls back to 'openai'.
    #[arg(long, default_value = None)]
    provider: Option<String>,

    /// Base URL to use for the specified provider.
    /// If not provided, tries to load it from the environment and,
    /// if not found in the environment, falls back to the default
    /// provider URL.
    #[arg(long, default_value = None)]
    base_url: Option<String>,

    /// Storage backend for sessions peristence. Allowed values: 'jsonl', 'sqlite'.
    /// Defaults to 'jsonl' (store session events as newline-separated JSON objects in a file)
    #[arg(long, default_value = None)]
    storage: Option<String>,

    /// Resume a previous session by id. If omitted, a new session is started.
    #[arg(long = "session-id", value_name = "ID")]
    session_id: Option<String>,

    /// Fork a previous session by id, creating a new session with cloned history.
    #[arg(long = "fork", value_name = "ID")]
    fork: Option<String>,

    /// Prompt to run in headless mode.
    #[arg(long, short, default_value = None)]
    prompt: Option<String>,

    /// Disable prompt caching (only applicable if Anthropic is the provider).
    #[arg(long, default_value_t = false)]
    no_prompt_cache: bool,

    /// Whether to print some debug/info messages when initializing the environment.
    /// This option has an effect in headless mode only.
    #[arg(long, short, default_value_t = false)]
    verbose: bool,
}

fn storage_choice(storage: Option<String>) -> AgentStorageChoice {
    match storage {
        Some(s) => match s.as_str() {
            "jsonl" => AgentStorageChoice::Jsonl,
            "sqlite" => AgentStorageChoice::Sqlite,
            _ => AgentStorageChoice::Jsonl,
        },
        None => AgentStorageChoice::Jsonl,
    }
}

async fn build_storage(
    storage: Option<String>,
) -> Result<Box<dyn microagents_storage::types::AgentStorage>, AgentError> {
    let st = storage_choice(storage);
    let builder = MicroAgentBuilder::<()>::new(ToolExecutionContext::new(()));
    let builder = builder
        .storage(st)
        .await
        .map_err(|e| AgentError::ClientInitFailed(e.to_string()))?;
    Ok(builder.storage)
}

async fn build_agent(
    provider: Option<String>,
    model: Option<String>,
    base_url: Option<String>,
    prompt_cache: bool,
    storage: Option<String>,
    skills: Vec<String>,
) -> Result<microagents_core::agent::MicroAgent<()>, AgentError> {
    let st = storage_choice(storage);
    let prov = match provider {
        None => infer_provider_from_env()?,
        Some(p) => SupportedProvider::from_str(&p)?,
    };
    let resolved_model = match model {
        Some(model) => model,
        None => prov.default_model()?.to_string(),
    };
    let base_builder = MicroAgentBuilder::<()>::new(ToolExecutionContext::new(()))
        .model(resolved_model)
        .provider(prov)?
        .storage(st)
        .await?
        .prompt_cache(prompt_cache)
        .load_agents_md()?
        .add_tool(Arc::new(tools::WriteTool))?
        .add_tool(Arc::new(tools::EditTool))?
        .add_tool(Arc::new(tools::ShellExecuteTool))?
        .add_tool(Arc::new(tools::SearchTool))?
        .add_tool(Arc::new(tools::ReadTool))?;
    let base_builder = if skills.is_empty() {
        base_builder.find_skills()?
    } else {
        let mut builder = base_builder;
        for skill in skills {
            builder = builder.add_skill(skill)?;
        }
        builder
    };
    let base_builder = if let Some(bu) = base_url {
        base_builder.base_url(bu)
    } else {
        base_builder
    };
    Ok(base_builder.build()?)
}

async fn fork_session(parent_id: &str, storage_opt: Option<String>) -> Result<String, AgentError> {
    let storage = build_storage(storage_opt).await?;
    let parent_events = storage.get_session(parent_id).await.map_err(|e| {
        AgentError::SessionLoadError(format!("Failed to load parent session: {}", e))
    })?;

    if parent_events.is_empty() {
        return Err(AgentError::SessionLoadError(format!(
            "Parent session {} has no events or does not exist",
            parent_id
        )));
    }

    let new_sid = uuid::Uuid::new_v4().to_string();

    for event in parent_events {
        if matches!(&event, microagents_events::AgentEventAny::SessionFork(_)) {
            continue;
        }
        let cloned_event = event.with_session_id(new_sid.clone());
        match cloned_event {
            microagents_events::AgentEventAny::SessionInit(init_event) => {
                storage.create_session(init_event).await.map_err(|e| {
                    AgentError::RunError(format!("Failed to write forked session init: {}", e))
                })?;
            }
            other => {
                storage.update_session(other).await.map_err(|e| {
                    AgentError::RunError(format!("Failed to write forked event: {}", e))
                })?;
            }
        }
    }

    let fork_event = microagents_events::SessionForkEvent {
        session_id: new_sid.clone(),
        parent_session_id: parent_id.to_string(),
        timestamp: chrono::Utc::now(),
    };
    storage
        .update_session(microagents_events::AgentEventAny::SessionFork(fork_event))
        .await
        .map_err(|e| AgentError::RunError(format!("Failed to write session fork event: {}", e)))?;

    Ok(new_sid)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let resolved_session_id = if let Some(parent_id) = &args.fork {
        if args.session_id.is_some() {
            return Err(
                "--fork-id and --session-id are mutually exclusive: only provide one of the two."
                    .into(),
            );
        }
        let new_sid = fork_session(parent_id, args.storage.clone()).await?;
        Some(new_sid)
    } else {
        args.session_id.clone()
    };

    if let Some(p) = args.prompt {
        initialize_environment(args.verbose).await?;
        let agent = build_agent(
            args.provider,
            args.model,
            args.base_url,
            !args.no_prompt_cache,
            args.storage,
            args.skill,
        )
        .await?;
        let mut stream = agent
            .run(p, resolved_session_id)
            .await
            .map_err(|e| e.to_string())?;
        while let Some(ev) = stream.next().await {
            match ev {
                Ok(e) => {
                    let jsonrpc = serde_json::to_string(&e.to_jsonrpc())?;
                    println!("{jsonrpc}");
                }
                Err(err) => {
                    let jsonrpc = serde_json::to_string(&serde_json::json!({
                        "jsonrpc": "2.0",
                        "id": 1,
                        "error": { "code": -32603, "message": "Internal Server Error", "data": &err.to_string() }
                    }))?;
                    println!("{jsonrpc}");
                    return Ok(());
                }
            }
        }
        return Ok(());
    }
    // when launching the TUI, you want the user
    // to know that some processing is taking place
    // and that things did not just randomly freeze
    initialize_environment(true).await?;
    println!("Launching TUI...");
    let initial_session = resolved_session_id.clone();
    let load_history_storage = args.storage.clone();
    tui::run_with_session(
        initial_session,
        move |prompt, session_id| {
            let prov_c = args.provider.clone();
            let model_c = args.model.clone();
            let skill_c = args.skill.clone();
            let storage_c = args.storage.clone();
            let base_url_c = args.base_url.clone();
            async move {
                // Building for TUI, verbose should always be true (as above)
                let agent = build_agent(
                    prov_c,
                    model_c,
                    base_url_c,
                    !args.no_prompt_cache,
                    storage_c,
                    skill_c,
                )
                .await?;
                agent.run(prompt, session_id).await
            }
        },
        move |session_id| {
            let storage_c = load_history_storage.clone();
            async move {
                let storage = build_storage(storage_c).await?;
                storage.get_session(&session_id).await.map_err(|e| {
                    microagents_core::types::AgentError::SessionLoadError(e.to_string())
                })
            }
        },
    )
    .await?;
    Ok(())
}
