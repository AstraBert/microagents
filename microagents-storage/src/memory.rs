use microagents_events::{AgentEventAny, SessionInitEvent, types::AgentEvent};
use std::{collections::HashMap, sync::Arc};
use tokio::sync::RwLock;

use crate::types::AgentStorage;

/// In-memory implementation of [`AgentStorage`].
///
/// All data is lost when the process exits.
#[derive(Debug)]
pub struct InMemoryAgentStorage {
    sessions: Arc<RwLock<HashMap<String, Vec<AgentEventAny>>>>,
}

impl Default for InMemoryAgentStorage {
    fn default() -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[async_trait::async_trait]
impl AgentStorage for InMemoryAgentStorage {
    async fn create_session(&self, event: SessionInitEvent) -> anyhow::Result<()> {
        let mut sessions = self.sessions.write().await;
        sessions.insert(
            event.session_id.clone(),
            vec![AgentEventAny::SessionInit(event)],
        );
        Ok(())
    }

    async fn update_session(&self, event: AgentEventAny) -> anyhow::Result<()> {
        let session_id = &event.session_id();
        let mut sessions = self.sessions.write().await;
        let session = sessions.get_mut(session_id);
        if let Some(s) = session {
            s.push(event);
            return Ok(());
        }
        Err(anyhow::anyhow!(
            "Could not find {session_id} among the registered sessions"
        ))
    }

    async fn get_session(&self, session_id: &str) -> anyhow::Result<Vec<AgentEventAny>> {
        let sessions = self.sessions.read().await;
        let session = sessions.get(session_id);
        if let Some(s) = session {
            let mut events = s.to_owned();
            events.sort_by_key(|a| a.timestamp());
            return Ok(events);
        }
        Err(anyhow::anyhow!(
            "Could not find {session_id} among the registered sessions"
        ))
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

    #[tokio::test]
    async fn test_default_init() {
        let memory = InMemoryAgentStorage::default();
        assert_eq!(memory.sessions.read().await.len(), 0);
    }

    #[tokio::test]
    async fn test_create_session() {
        let memory = InMemoryAgentStorage::default();
        memory
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
        let sessions = memory.sessions.read().await;
        assert!(sessions.get("1").is_some_and(|v| {
            v.len() == 1
                && v.first()
                    .is_some_and(|f| f.clone().to_jsonrpc().method == "session.init")
        }));
    }

    #[tokio::test]
    async fn test_create_update_get_session() {
        let memory = InMemoryAgentStorage::default();
        memory
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
        memory
            .update_session(AgentEventAny::UserPromptSubmit(UserPromptSubmitEvent {
                prompt: "hello".to_string(),
                session_id: "1".to_string(),
                turn_id: "t1".to_string(),
                timestamp: Utc::now(),
            }))
            .await
            .expect("Should be able to update memory");
        memory
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
        memory
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
        let events = memory
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
