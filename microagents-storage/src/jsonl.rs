use microagents_events::types::{AgentEvent, JsonRpcNotification};
use microagents_events::{AgentEventAny, SessionInitEvent};
use std::{path::PathBuf, sync::OnceLock};
use tokio::fs::OpenOptions;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use crate::types::AgentStorage;

/// Global default directory for JSONL session files.
pub static JSONL_SESSION_STORAGE: OnceLock<PathBuf> = OnceLock::new();

/// Return the default JSONL storage directory (`~/.microagents/sessions`).
pub fn jsonl_session_storage() -> &'static PathBuf {
    JSONL_SESSION_STORAGE.get_or_init(|| {
        dirs::home_dir()
            .expect("could not determine home directory")
            .join(".microagents")
            .join("sessions")
    })
}

/// JSON Lines file-based implementation of [`AgentStorage`].
///
/// Each session is stored as a separate `.jsonl` file under `jsonl_path`.
#[derive(Debug)]
pub struct JsonlAgentStorage {
    /// Directory where session `.jsonl` files are written.
    pub jsonl_path: PathBuf,
}

impl Default for JsonlAgentStorage {
    fn default() -> Self {
        Self {
            jsonl_path: jsonl_session_storage().to_owned(),
        }
    }
}

impl JsonlAgentStorage {
    /// Create a new [`JsonlAgentStorage`].
    ///
    /// If `jsonl_path` is `None`, the default directory from [`jsonl_session_storage`] is used.
    pub fn new(jsonl_path: Option<PathBuf>) -> Self {
        Self {
            jsonl_path: jsonl_path.unwrap_or(jsonl_session_storage().to_owned()),
        }
    }

    async fn ensure_sessions_dir(&self) -> anyhow::Result<()> {
        if self.jsonl_path.is_dir() {
            return Ok(());
        }
        tokio::fs::create_dir_all(&self.jsonl_path).await?;
        Ok(())
    }
}

#[async_trait::async_trait]
impl AgentStorage for JsonlAgentStorage {
    async fn create_session(&self, event: SessionInitEvent) -> anyhow::Result<()> {
        self.ensure_sessions_dir().await?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.jsonl_path.join(format!("{}.jsonl", event.session_id)))
            .await?;
        let event_json = serde_json::to_string(&event.to_jsonrpc())?;
        file.write_all(format!("{}\n", event_json).as_bytes())
            .await?;
        Ok(())
    }

    async fn update_session(&self, event: AgentEventAny) -> anyhow::Result<()> {
        self.ensure_sessions_dir().await?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(
                self.jsonl_path
                    .join(format!("{}.jsonl", event.session_id())),
            )
            .await?;
        file.write_all(format!("{}\n", serde_json::to_string(&event.to_jsonrpc())?).as_bytes())
            .await?;
        Ok(())
    }

    async fn get_session(&self, session_id: &str) -> anyhow::Result<Vec<AgentEventAny>> {
        self.ensure_sessions_dir().await?;
        let mut file = OpenOptions::new()
            .read(true)
            .open(self.jsonl_path.join(format!("{session_id}.jsonl")))
            .await?;
        let mut buf = String::new();
        file.read_to_string(&mut buf).await?;
        let mut events = vec![];
        let mut i = 0;
        for line in buf.lines() {
            i += 1;
            let jsrpc: JsonRpcNotification = match serde_json::from_str(line.trim_end_matches("\n"))
            {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("Corrupted line {:?}. Error detail: {}", i, e);
                    continue;
                }
            };
            let event = AgentEventAny::try_from(jsrpc)?;
            events.push(event);
        }

        events.sort_by_key(|a| a.timestamp());

        Ok(events)
    }
}

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use microagents_events::{
        AssistantMessagePart, AssistantResponseEvent, SessionStopEvent, TextPart, Usage,
        UserPromptSubmitEvent,
    };

    use super::*;

    #[test]
    fn test_default_init() {
        let jsonl = JsonlAgentStorage::default();
        assert_eq!(jsonl.jsonl_path, jsonl_session_storage().to_owned());
    }

    #[tokio::test]
    async fn test_create_session() {
        let tmp = tempfile::tempdir().unwrap();
        let jsonl = JsonlAgentStorage::new(Some(tmp.path().to_path_buf()));
        jsonl
            .create_session(SessionInitEvent {
                session_id: "1".to_string(),
                model: "gpt-5.5".into(),
                provider: "openai".into(),
                system: "you are a helpful assistant".into(),
                init_type: microagents_events::SessionInitType::Start,
                timestamp: Utc::now(),
            })
            .await
            .expect("Should be able to create a session");
        let content = tokio::fs::read_to_string(tmp.path().join("1.jsonl"))
            .await
            .expect("Should be able to read file");
        let mut events = vec![];
        for line in content.lines() {
            let jsrpc: JsonRpcNotification =
                serde_json::from_str(line).expect("Should serialize correctly");
            let event = AgentEventAny::try_from(jsrpc).expect("Should convert to agent event");
            events.push(event);
        }
        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0].clone().to_jsonrpc().method,
            "session.init".to_string()
        );
    }

    #[tokio::test]
    async fn test_create_update_get_session() {
        let tmp = tempfile::tempdir().unwrap();
        let jsonl = JsonlAgentStorage::new(Some(tmp.path().to_path_buf()));
        jsonl
            .create_session(SessionInitEvent {
                session_id: "1".to_string(),
                model: "gpt-5.5".into(),
                provider: "openai".into(),
                system: "you are a helpful assistant".into(),
                init_type: microagents_events::SessionInitType::Start,
                timestamp: Utc::now(),
            })
            .await
            .expect("Should be able to create a session");
        jsonl
            .update_session(AgentEventAny::UserPromptSubmit(UserPromptSubmitEvent {
                prompt: "hello".to_string(),
                session_id: "1".to_string(),
                turn_id: "t1".to_string(),
                timestamp: Utc::now(),
            }))
            .await
            .expect("Should be able to update memory");
        jsonl
            .update_session(AgentEventAny::AssistantResponse(AssistantResponseEvent {
                session_id: "1".to_string(),
                turn_id: "t1".to_string(),
                content: vec![AssistantMessagePart::Text(TextPart {
                    text: "hello".to_string(),
                })],
                timestamp: Utc::now(),
            }))
            .await
            .expect("Should be able to update memory");
        jsonl
            .update_session(AgentEventAny::SessionStop(SessionStopEvent {
                session_id: "1".to_string(),
                result: Some(vec![AssistantMessagePart::Text(TextPart {
                    text: "hello".to_string(),
                })]),
                error: None,
                success: true,
                timestamp: Utc::now(),
                usage: Usage::default(),
                incomplete_tasks: None,
            }))
            .await
            .expect("Should be able to update memory");
        let events = jsonl
            .get_session("1")
            .await
            .expect("Should be able to get the session");
        assert_eq!(events.len(), 4);
        assert_eq!(events[0].to_jsonrpc().method, "session.init".to_string());
        assert_eq!(
            events[1].to_jsonrpc().method,
            "user.prompt.submit".to_string()
        );
        assert_eq!(
            events[2].to_jsonrpc().method,
            "assistant.response".to_string()
        );
        assert_eq!(events[3].to_jsonrpc().method, "session.stop".to_string());
    }
}
