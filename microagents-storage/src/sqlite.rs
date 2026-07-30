use crate::types::AgentStorage;
use microagents_events::{
    AgentEventAny, SessionInitEvent,
    types::{AgentEvent, JsonRpcNotification},
};
use std::{
    path::PathBuf,
    sync::{Arc, OnceLock},
    time::SystemTime,
};
use tokio::sync::Mutex;
use tokio_rusqlite::Connection;
use tokio_rusqlite::rusqlite;

/// Global default path for the SQLite sessions database.
pub static SQLITE_SESSION_STORAGE: OnceLock<PathBuf> = OnceLock::new();

/// Return the default SQLite database path (`~/.microagents/sessions.db`).
pub fn sqlite_session_storage() -> &'static PathBuf {
    SQLITE_SESSION_STORAGE.get_or_init(|| {
        dirs::home_dir()
            .expect("could not determine home directory")
            .join(".microagents")
            .join("sessions.db")
    })
}

/// SQLite-backed implementation of [`AgentStorage`].
#[derive(Debug, Clone)]
pub struct SqliteAgentStorage {
    connection: Arc<Mutex<Connection>>,
}

impl SqliteAgentStorage {
    /// Open (or create) the SQLite database at `db_path` and ensure the events table exists.
    ///
    /// If `db_path` is `None`, the default path from [`sqlite_session_storage`] is used.
    pub async fn new(db_path: Option<&PathBuf>) -> anyhow::Result<Self> {
        let path = db_path.unwrap_or(sqlite_session_storage());
        if let Some(parent) = std::path::Path::new(&path).parent()
            && !parent.exists()
        {
            std::fs::create_dir_all(parent)?;
        }
        let connection = Arc::new(Mutex::new(Connection::open(path).await?));
        let storage = Self { connection };
        storage.ensure_table_and_idx().await?;
        Ok(storage)
    }

    async fn ensure_table_and_idx(&self) -> anyhow::Result<()> {
        self.connection
            .lock()
            .await
            .call(|conn| -> Result<(), tokio_rusqlite::rusqlite::Error> {
                conn.execute_batch(
                    r#"
                    PRAGMA journal_mode = WAL;
                    PRAGMA synchronous = NORMAL;
                    "#,
                )?;

                // useful for future migrations
                let version: i64 = conn.query_row("PRAGMA user_version", [], |r| r.get(0))?;

                if version < 1 {
                    conn.execute_batch(
                        r#"
                    CREATE TABLE IF NOT EXISTS events (
                        id          INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id  TEXT    NOT NULL,
                        payload     TEXT    NOT NULL,
                        created_at  INTEGER NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_events_session_id ON events(session_id);
                    PRAGMA user_version = 1;
                    "#,
                    )?;
                }
                Ok(())
            })
            .await?;
        Ok(())
    }
}

#[async_trait::async_trait]
impl AgentStorage for SqliteAgentStorage {
    async fn create_session(&self, event: SessionInitEvent) -> anyhow::Result<()> {
        let session_id = event.session_id.clone();
        let json_event = serde_json::to_string(&event.to_jsonrpc())?;
        let now = now_millis()?;

        self.connection
            .lock()
            .await
            .call(move |conn| -> Result<(), tokio_rusqlite::rusqlite::Error> {
                conn.execute(
                    "INSERT INTO events (session_id, payload, created_at) VALUES (?1, ?2, ?3)",
                    rusqlite::params![session_id, json_event, now],
                )?;
                Ok(())
            })
            .await?;
        Ok(())
    }

    async fn update_session(&self, event: AgentEventAny) -> anyhow::Result<()> {
        let session_id = event.session_id();
        let json_event = serde_json::to_string(&event.to_jsonrpc())?;
        let now = now_millis()?;

        self.connection
            .lock()
            .await
            .call(move |conn| -> Result<(), tokio_rusqlite::rusqlite::Error> {
                conn.execute(
                    "INSERT INTO events (session_id, payload, created_at) VALUES (?1, ?2, ?3)",
                    rusqlite::params![session_id, json_event, now],
                )?;
                Ok(())
            })
            .await?;
        Ok(())
    }

    async fn get_session(&self, session_id: &str) -> anyhow::Result<Vec<AgentEventAny>> {
        let session_id = session_id.to_string();

        let rows =
            self.connection
                .lock()
                .await
                .call(
                    move |conn| -> Result<Vec<(isize, String)>, tokio_rusqlite::rusqlite::Error> {
                        let mut stmt = conn.prepare(
                            "SELECT id, payload FROM events WHERE session_id = ?1 ORDER BY id ASC",
                        )?;
                        let rows = stmt
                    .query_map([&session_id], |row| {
                        Ok((row.get::<_, isize>(0)?, row.get::<_, String>(1)?))
                    })?
                    .collect::<Result<Vec<(isize, String)>, tokio_rusqlite::rusqlite::Error>>()?;
                        Ok(rows)
                    },
                )
                .await?;

        let mut events: Vec<AgentEventAny> = Vec::with_capacity(rows.len());
        for (_, payload) in rows {
            let jrpc: JsonRpcNotification = serde_json::from_str(&payload)
                .map_err(|e| anyhow::anyhow!("Invalid JSON payload in events table: {e}"))?;
            let event = AgentEventAny::try_from(jrpc)
                .map_err(|e| anyhow::anyhow!("Invalid event payload in events table: {e}"))?;
            events.push(event);
        }
        events.sort_by_key(|a| a.timestamp());
        Ok(events)
    }
}

fn now_millis() -> anyhow::Result<i64> {
    Ok(SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)?
        .as_millis()
        .try_into()?)
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
        let tmp = tempfile::tempdir().unwrap();
        let db_path = tmp.path().join("sessions.db");
        let result = SqliteAgentStorage::new(Some(&db_path)).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_create_session() {
        let tmp = tempfile::tempdir().unwrap();
        let db_path = tmp.path().join("test.db");
        let sql = SqliteAgentStorage::new(Some(&db_path))
            .await
            .expect("Should be able to open agent store");
        sql.create_session(SessionInitEvent {
            session_id: "1".to_string(),
            model: "gpt-5.5".into(),
            provider: "openai".into(),
            system: "you are a helpful assistant".into(),
            init_type: microagents_events::SessionInitType::Start,
            timestamp: Utc::now(),
        })
        .await
        .expect("Should be able to create a session");
        let rows =
            sql.connection
                .lock()
                .await
                .call(
                    move |conn| -> Result<Vec<(isize, String)>, tokio_rusqlite::rusqlite::Error> {
                        let mut stmt = conn.prepare(
                            "SELECT id, payload FROM events WHERE session_id = ?1 ORDER BY id ASC",
                        )?;
                        let rows = stmt
                    .query_map(["1"], |row| {
                        Ok((row.get::<_, isize>(0)?, row.get::<_, String>(1)?))
                    })?
                    .collect::<Result<Vec<(isize, String)>, tokio_rusqlite::rusqlite::Error>>()?;
                        Ok(rows)
                    },
                )
                .await
                .expect("Should be able to perform sql operation");

        let events: Vec<AgentEventAny> = rows
            .into_iter()
            .map(|(_, payload)| {
                let jrpc: JsonRpcNotification = serde_json::from_str(&payload).unwrap();
                AgentEventAny::try_from(jrpc).unwrap()
            })
            .collect();
        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0].clone().to_jsonrpc().method,
            "session.init".to_string()
        );
        // Connection is dropped when `sql` goes out of scope, then tmp dir is cleaned up
    }

    #[tokio::test]
    async fn test_create_update_get_session() {
        let tmp = tempfile::tempdir().unwrap();
        let db_path = tmp.path().join("test.db");
        let sql = SqliteAgentStorage::new(Some(&db_path))
            .await
            .expect("Should be able to create sqlite store");
        sql.create_session(SessionInitEvent {
            session_id: "1".to_string(),
            model: "gpt-5.5".into(),
            provider: "openai".into(),
            system: "you are a helpful assistant".into(),
            init_type: microagents_events::SessionInitType::Start,
            timestamp: Utc::now(),
        })
        .await
        .expect("Should be able to create a session");
        sql.update_session(AgentEventAny::UserPromptSubmit(UserPromptSubmitEvent {
            prompt: "hello".to_string(),
            session_id: "1".to_string(),
            turn_id: "t1".to_string(),
            timestamp: Utc::now(),
        }))
        .await
        .expect("Should be able to update memory");
        sql.update_session(AgentEventAny::AssistantResponse(AssistantResponseEvent {
            session_id: "1".to_string(),
            turn_id: "t1".to_string(),
            content: vec![AssistantMessagePart::Text(TextPart {
                text: "hello".to_string(),
            })],
            timestamp: Utc::now(),
        }))
        .await
        .expect("Should be able to update memory");
        sql.update_session(AgentEventAny::SessionStop(SessionStopEvent {
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
        let events = sql
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
        // Connection is dropped when `sql` goes out of scope, then tmp dir is cleaned up
    }
}
