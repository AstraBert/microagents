pub mod types;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use std::{convert::TryFrom, fmt};

use crate::types::{AgentEvent, AgentEventError, JsonRpcNotification, ToolResult};

pub const EVENTS_PROTOCOL_VERSION: &str = "0.3.0";

/// Indicates whether a session is being started fresh or resumed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum SessionInitType {
    /// Start a new session.
    Start,
    /// Resume an existing session.
    Resume,
}

impl fmt::Display for SessionInitType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Start => write!(f, "start"),
            Self::Resume => write!(f, "resume"),
        }
    }
}

impl TryFrom<&str> for SessionInitType {
    type Error = AgentEventError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value.to_lowercase().as_str() {
            "start" => Ok(Self::Start),
            "resume" => Ok(Self::Resume),
            _ => Err(AgentEventError::InvalidFieldType(format!(
                "No init type with message: {}",
                value
            ))),
        }
    }
}

/// Type of content delta in a stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum DeltaType {
    /// Regular text content.
    Text,
    /// Model thinking or reasoning content.
    Thinking,
}

impl fmt::Display for DeltaType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Text => write!(f, "text"),
            Self::Thinking => write!(f, "thinking"),
        }
    }
}

/// Status enum for an agent task
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum TaskStatus {
    Queued,
    InProgress,
    Done,
}

impl fmt::Display for TaskStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Queued => write!(f, "queued"),
            Self::InProgress => write!(f, "in_progress"),
            Self::Done => write!(f, "done"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Copy, Default)]
pub struct Usage {
    pub latency: i64,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cache_write_tokens: u32,
    pub cache_read_tokens: u32,
}

/// Event emitted when a session is initialized.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionInitEvent {
    pub session_id: String,
    pub model: String,
    pub provider: String,
    pub system: String,
    /// Either `'new'` or `'resume'`.
    pub init_type: SessionInitType,
    pub timestamp: DateTime<Utc>,
}

/// Event emitted when a session stops.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionStopEvent {
    pub session_id: String,
    pub success: bool,
    pub result: Option<Vec<AssistantMessagePart>>,
    pub error: Option<String>,
    pub timestamp: DateTime<Utc>,
    pub incomplete_tasks: Option<Vec<String>>,
    pub usage: Usage,
}

/// Event emitted when the user submits a prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPromptSubmitEvent {
    pub session_id: String,
    pub turn_id: String,
    pub prompt: String,
    pub timestamp: DateTime<Utc>,
}

/// Event emitted for each delta in a streaming response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamDeltaEvent {
    pub session_id: String,
    pub turn_id: String,
    pub delta: String,
    pub delta_type: DeltaType,
    pub timestamp: DateTime<Utc>,
}

/// Event emitted when a tool is called.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallEvent {
    pub session_id: String,
    pub turn_id: String,
    pub name: String,
    pub input: Value,
    pub timestamp: DateTime<Utc>,
}

/// Event emitted when a tool returns a result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResultEvent {
    pub session_id: String,
    pub turn_id: String,
    /// Tool execution result. [`Value`] implements `From<ToolResult>`.
    pub result: ToolResult,
    pub tool_call_id: String,
    pub timestamp: DateTime<Utc>,
}

/// Event emitted when a skill is loaded.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillLoadEvent {
    pub session_id: String,
    pub turn_id: String,
    pub skill_name: String,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct TextPart {
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct ThinkingPart {
    pub thinking: String,
    pub signature: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct ToolCallPart {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
#[serde(untagged)]
pub enum AssistantMessagePart {
    Text(TextPart),
    Thinking(ThinkingPart),
    ToolCall(ToolCallPart),
}

/// Event emitted when the assistant produces a complete response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantResponseEvent {
    pub session_id: String,
    pub turn_id: String,
    pub content: Vec<AssistantMessagePart>,
    pub timestamp: DateTime<Utc>,
}

/// Event emitted when the event creates, executes and finishes a task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskEvent {
    pub session_id: String,
    pub turn_id: String,
    pub task_name: String,
    pub task_status: TaskStatus,
    pub timestamp: DateTime<Utc>,
}

/// Event emitted when a session is forked from a parent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionForkEvent {
    pub session_id: String,
    pub parent_session_id: String,
    pub timestamp: DateTime<Utc>,
}

impl AgentEvent for SessionForkEvent {
    fn to_jsonrpc(&self) -> JsonRpcNotification {
        serialize_to_jsonrpc("session.fork", self)
    }

    fn session_id(&self) -> String {
        self.session_id.clone()
    }
}

fn serialize_to_jsonrpc<T: Serialize>(method: &str, params_struct: &T) -> JsonRpcNotification {
    let params_value =
        serde_json::to_value(params_struct).expect("Event params serialization failed");
    let params = match params_value {
        Value::Object(m) => m,
        other => panic!(
            "Serialized event params must be a JSON Object, found: {:?}",
            other
        ),
    };
    JsonRpcNotification {
        jsonrpc: "2.0".to_string(),
        method: method.to_string(),
        params,
    }
}

impl AgentEvent for SessionInitEvent {
    fn to_jsonrpc(&self) -> JsonRpcNotification {
        serialize_to_jsonrpc("session.init", self)
    }

    fn session_id(&self) -> String {
        self.session_id.clone()
    }
}

impl AgentEvent for SessionStopEvent {
    fn to_jsonrpc(&self) -> JsonRpcNotification {
        serialize_to_jsonrpc("session.stop", self)
    }

    fn session_id(&self) -> String {
        self.session_id.clone()
    }
}

impl AgentEvent for UserPromptSubmitEvent {
    fn to_jsonrpc(&self) -> JsonRpcNotification {
        serialize_to_jsonrpc("user.prompt.submit", self)
    }

    fn session_id(&self) -> String {
        self.session_id.clone()
    }
}

impl AgentEvent for StreamDeltaEvent {
    fn to_jsonrpc(&self) -> JsonRpcNotification {
        serialize_to_jsonrpc("stream.delta", self)
    }

    fn session_id(&self) -> String {
        self.session_id.clone()
    }
}

impl AgentEvent for ToolCallEvent {
    fn to_jsonrpc(&self) -> JsonRpcNotification {
        serialize_to_jsonrpc("tool.call", self)
    }

    fn session_id(&self) -> String {
        self.session_id.clone()
    }
}

impl AgentEvent for ToolResultEvent {
    fn to_jsonrpc(&self) -> JsonRpcNotification {
        serialize_to_jsonrpc("tool.result", self)
    }

    fn session_id(&self) -> String {
        self.session_id.clone()
    }
}

impl AgentEvent for SkillLoadEvent {
    fn to_jsonrpc(&self) -> JsonRpcNotification {
        serialize_to_jsonrpc("skill.load", self)
    }

    fn session_id(&self) -> String {
        self.session_id.clone()
    }
}

impl AgentEvent for AssistantResponseEvent {
    fn to_jsonrpc(&self) -> JsonRpcNotification {
        serialize_to_jsonrpc("assistant.response", self)
    }

    fn session_id(&self) -> String {
        self.session_id.clone()
    }
}

impl AgentEvent for TaskEvent {
    fn to_jsonrpc(&self) -> JsonRpcNotification {
        serialize_to_jsonrpc("assistant.task", self)
    }

    fn session_id(&self) -> String {
        self.session_id.clone()
    }
}

/// A sum type wrapping any agent event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "method", content = "params", rename_all = "snake_case")]
#[non_exhaustive]
pub enum AgentEventAny {
    #[serde(rename = "session.init")]
    SessionInit(SessionInitEvent),
    #[serde(rename = "session.stop")]
    SessionStop(SessionStopEvent),
    #[serde(rename = "stream.delta")]
    StreamDelta(StreamDeltaEvent),
    #[serde(rename = "tool.call")]
    ToolCall(ToolCallEvent),
    #[serde(rename = "tool.result")]
    ToolResult(ToolResultEvent),
    #[serde(rename = "assistant.response")]
    AssistantResponse(AssistantResponseEvent),
    #[serde(rename = "skill.load")]
    SkillLoad(SkillLoadEvent),
    #[serde(rename = "user.prompt.submit")]
    UserPromptSubmit(UserPromptSubmitEvent),
    #[serde(rename = "assistant.task")]
    Task(TaskEvent),
    #[serde(rename = "session.fork")]
    SessionFork(SessionForkEvent),
}

impl AgentEventAny {
    pub fn timestamp(&self) -> DateTime<Utc> {
        match self {
            Self::SessionInit(s) => s.timestamp,
            Self::AssistantResponse(s) => s.timestamp,
            Self::SessionStop(s) => s.timestamp,
            Self::SkillLoad(s) => s.timestamp,
            Self::StreamDelta(s) => s.timestamp,
            Self::UserPromptSubmit(s) => s.timestamp,
            Self::ToolCall(s) => s.timestamp,
            Self::ToolResult(s) => s.timestamp,
            Self::Task(s) => s.timestamp,
            Self::SessionFork(s) => s.timestamp,
        }
    }

    pub fn with_session_id(mut self, sid: String) -> Self {
        match &mut self {
            Self::SessionInit(s) => s.session_id = sid,
            Self::AssistantResponse(s) => s.session_id = sid,
            Self::SessionStop(s) => s.session_id = sid,
            Self::SkillLoad(s) => s.session_id = sid,
            Self::StreamDelta(s) => s.session_id = sid,
            Self::UserPromptSubmit(s) => s.session_id = sid,
            Self::ToolCall(s) => s.session_id = sid,
            Self::ToolResult(s) => s.session_id = sid,
            Self::Task(s) => s.session_id = sid,
            Self::SessionFork(s) => s.session_id = sid,
        }
        self
    }
}

impl AgentEvent for AgentEventAny {
    fn to_jsonrpc(&self) -> JsonRpcNotification {
        match self {
            Self::SessionInit(e) => e.to_jsonrpc(),
            Self::SessionStop(e) => e.to_jsonrpc(),
            Self::StreamDelta(e) => e.to_jsonrpc(),
            Self::ToolCall(e) => e.to_jsonrpc(),
            Self::ToolResult(e) => e.to_jsonrpc(),
            Self::AssistantResponse(e) => e.to_jsonrpc(),
            Self::SkillLoad(e) => e.to_jsonrpc(),
            Self::UserPromptSubmit(e) => e.to_jsonrpc(),
            Self::Task(e) => e.to_jsonrpc(),
            Self::SessionFork(e) => e.to_jsonrpc(),
        }
    }

    fn session_id(&self) -> String {
        match self {
            Self::AssistantResponse(s) => s.session_id.clone(),
            Self::SessionInit(s) => s.session_id.clone(),
            Self::SessionStop(s) => s.session_id.clone(),
            Self::StreamDelta(s) => s.session_id.clone(),
            Self::SkillLoad(s) => s.session_id.clone(),
            Self::ToolCall(s) => s.session_id.clone(),
            Self::ToolResult(s) => s.session_id.clone(),
            Self::UserPromptSubmit(s) => s.session_id.clone(),
            Self::Task(s) => s.session_id.clone(),
            Self::SessionFork(s) => s.session_id.clone(),
        }
    }
}

impl TryFrom<JsonRpcNotification> for AgentEventAny {
    type Error = AgentEventError;

    fn try_from(value: JsonRpcNotification) -> Result<Self, Self::Error> {
        let mut map = serde_json::Map::new();
        map.insert("method".to_string(), Value::String(value.method.clone()));
        map.insert("params".to_string(), Value::Object(value.params));

        serde_json::from_value::<AgentEventAny>(Value::Object(map)).map_err(|err| {
            let msg = err.to_string();
            let field = msg.split('`').nth(1).unwrap_or("unknown").to_string();
            if msg.contains("unknown variant") && field == value.method {
                AgentEventError::UnknownMethod(value.method)
            } else if msg.contains("missing field") {
                AgentEventError::MissingField(field)
            } else {
                AgentEventError::InvalidFieldType(field)
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn session_init_type_from_value() {
        let start: SessionInitType = serde_json::from_value(Value::from("Start"))
            .expect("Should be able to convert to SessionInitType");
        let resume: SessionInitType = serde_json::from_value(Value::from("Resume"))
            .expect("Should be able to convert to SessionInitType");
        assert!(matches!(start, SessionInitType::Start));
        assert!(matches!(resume, SessionInitType::Resume));
    }

    #[test]
    fn session_init_type_from_str_ok() {
        assert!(matches!(
            SessionInitType::try_from("start"),
            Ok(SessionInitType::Start)
        ));
        assert!(matches!(
            SessionInitType::try_from("resume"),
            Ok(SessionInitType::Resume)
        ));
    }

    #[test]
    fn session_init_type_from_str_err() {
        let err = SessionInitType::try_from("unknown").unwrap_err();
        assert!(matches!(err, AgentEventError::InvalidFieldType(_)));
        assert!(
            err.to_string()
                .contains("No init type with message: unknown")
        );
    }

    #[test]
    fn session_init_event_to_jsonrpc() {
        let event = SessionInitEvent {
            session_id: "s1".into(),
            model: "gpt-4".into(),
            provider: "openai".into(),
            system: "sys".into(),
            init_type: SessionInitType::Start,
            timestamp: Utc::now(),
        };
        let rpc = event.to_jsonrpc();
        assert_eq!(rpc.method, "session.init");
        assert_eq!(rpc.params.get("session_id"), Some(&Value::from("s1")));
        assert_eq!(rpc.params.get("model"), Some(&Value::from("gpt-4")));
        assert_eq!(rpc.params.get("provider"), Some(&Value::from("openai")));
        assert_eq!(rpc.params.get("system"), Some(&Value::from("sys")));
        assert_eq!(rpc.params.get("init_type"), Some(&Value::from("Start")));
    }

    #[test]
    fn session_stop_event_to_jsonrpc() {
        let event = SessionStopEvent {
            session_id: "s1".into(),
            success: true,
            result: Some(vec![AssistantMessagePart::Text(TextPart {
                text: "done".to_string(),
            })]),
            error: None,
            timestamp: Utc::now(),
            usage: Usage::default(),
            incomplete_tasks: None,
        };
        let rpc = event.to_jsonrpc();
        assert_eq!(rpc.method, "session.stop");
        assert_eq!(rpc.params.get("success"), Some(&Value::from(true)));
        assert_eq!(
            rpc.params.get("result"),
            Some(&Value::from(vec![json!({"text": "done"})]))
        );
        assert_eq!(rpc.params.get("error"), Some(&Value::Null));
        assert_eq!(rpc.params.get("incomplete_tasks"), Some(&Value::Null));
        assert_eq!(
            rpc.params.get("usage"),
            Some(
                &serde_json::to_value(Usage::default())
                    .expect("Should be able to convert to value")
            )
        );
    }

    #[test]
    fn user_prompt_submit_event_to_jsonrpc() {
        let event = UserPromptSubmitEvent {
            session_id: "s1".into(),
            turn_id: "t1".into(),
            prompt: "hello".into(),
            timestamp: Utc::now(),
        };
        let rpc = event.to_jsonrpc();
        assert_eq!(rpc.method, "user.prompt.submit");
        assert_eq!(rpc.params.get("prompt"), Some(&Value::from("hello")));
    }

    #[test]
    fn stream_delta_event_to_jsonrpc() {
        let event = StreamDeltaEvent {
            session_id: "s1".into(),
            turn_id: "t1".into(),
            delta: "world".into(),
            delta_type: DeltaType::Thinking,
            timestamp: Utc::now(),
        };
        let rpc = event.to_jsonrpc();
        assert_eq!(rpc.method, "stream.delta");
        assert_eq!(rpc.params.get("delta"), Some(&Value::from("world")));
        assert_eq!(rpc.params.get("delta_type"), Some(&Value::from("Thinking")));
    }

    #[test]
    fn tool_call_event_to_jsonrpc() {
        let event = ToolCallEvent {
            session_id: "s1".into(),
            turn_id: "t1".into(),
            name: "read".into(),
            input: json!({"path": "/tmp"}),
            timestamp: Utc::now(),
        };
        let rpc = event.to_jsonrpc();
        assert_eq!(rpc.method, "tool.call");
        assert_eq!(rpc.params.get("name"), Some(&Value::from("read")));
        assert_eq!(rpc.params.get("input"), Some(&json!({"path": "/tmp"})));
    }

    #[test]
    fn tool_result_event_to_jsonrpc() {
        let event = ToolResultEvent {
            session_id: "s1".into(),
            turn_id: "t1".into(),
            result: ToolResult::Ok("ok".into()),
            tool_call_id: "tc1".into(),
            timestamp: Utc::now(),
        };
        let rpc = event.to_jsonrpc();
        assert_eq!(rpc.method, "tool.result");
        assert_eq!(rpc.params.get("tool_call_id"), Some(&Value::from("tc1")));
        assert_eq!(rpc.params.get("result"), Some(&json!({"Ok": "ok"})));
    }

    #[test]
    fn skill_load_event_to_jsonrpc() {
        let event = SkillLoadEvent {
            session_id: "s1".into(),
            turn_id: "t1".into(),
            skill_name: "coding".into(),
            timestamp: Utc::now(),
        };
        let rpc = event.to_jsonrpc();
        assert_eq!(rpc.method, "skill.load");
        assert_eq!(rpc.params.get("skill_name"), Some(&Value::from("coding")));
    }

    #[test]
    fn assistant_response_event_to_jsonrpc() {
        let event = AssistantResponseEvent {
            session_id: "s1".into(),
            turn_id: "t1".into(),
            content: vec![AssistantMessagePart::Text(TextPart {
                text: "hi".to_string(),
            })],
            timestamp: Utc::now(),
        };
        let rpc = event.to_jsonrpc();
        assert_eq!(rpc.method, "assistant.response");
        let content = rpc.params.get("content");
        match content {
            None => panic!("`content` should be a key in the rpc message"),
            Some(v) => match v.as_array() {
                None => panic!("`content` should be an array"),
                Some(values) => {
                    let first = values.first().expect("Should have one element");
                    match first.as_object() {
                        None => panic!("Element of `content` should be an object"),
                        Some(v) => {
                            assert!(v.get("text").is_some_and(|g| g == &Value::from("hi")));
                        }
                    }
                }
            },
        }
    }

    #[test]
    fn task_event_to_jsonrpc() {
        let event = TaskEvent {
            session_id: "s1".into(),
            turn_id: "t1".into(),
            task_status: TaskStatus::InProgress,
            task_name: "test".into(),
            timestamp: Utc::now(),
        };
        let rpc = event.to_jsonrpc();
        assert_eq!(rpc.method, "assistant.task");
        assert_eq!(rpc.params.get("task_name"), Some(&Value::from("test")));
        assert_eq!(
            rpc.params.get("task_status"),
            Some(&serde_json::to_value(TaskStatus::InProgress).unwrap())
        );
    }

    #[test]
    fn agent_event_any_session_id() {
        let event = AgentEventAny::SessionInit(SessionInitEvent {
            session_id: "sid".into(),
            model: "m".into(),
            provider: "p".into(),
            system: "s".into(),
            init_type: SessionInitType::Start,
            timestamp: Utc::now(),
        });
        assert_eq!(event.session_id(), "sid");
    }

    #[test]
    fn agent_event_any_to_jsonrpc_roundtrip() {
        let event = AgentEventAny::UserPromptSubmit(UserPromptSubmitEvent {
            session_id: "s1".into(),
            turn_id: "t1".into(),
            prompt: "p".into(),
            timestamp: Utc::now(),
        });
        let rpc = event.to_jsonrpc();
        assert_eq!(rpc.method, "user.prompt.submit");
        assert_eq!(rpc.params.get("session_id"), Some(&Value::from("s1")));
    }

    #[test]
    fn try_from_jsonrpc_session_init_ok() {
        let rpc = JsonRpcNotification::builder()
            .method("session.init".into())
            .add_param("session_id".into(), Value::from("s1"))
            .add_param("model".into(), Value::from("gpt-4"))
            .add_param("provider".into(), Value::from("openai"))
            .add_param("system".into(), Value::from("sys"))
            .add_param("init_type".into(), Value::from("Resume"))
            .add_param("timestamp".into(), {
                let tms = Utc::now();
                serde_json::to_value(tms).expect("Should convert to value")
            });
        let any = AgentEventAny::try_from(rpc).unwrap();
        assert!(
            matches!(any, AgentEventAny::SessionInit(ref e) if e.session_id == "s1" && matches!(e.init_type, SessionInitType::Resume))
        );
    }

    #[test]
    fn try_from_jsonrpc_session_init_missing_field() {
        let rpc = JsonRpcNotification::builder()
            .method("session.init".into())
            .add_param("session_id".into(), Value::from("s1"))
            .add_param("timestamp".into(), {
                let tms = Utc::now();
                serde_json::to_value(tms).expect("Should convert to value")
            });
        let err = AgentEventAny::try_from(rpc).unwrap_err();
        assert!(matches!(err, AgentEventError::MissingField(_)));
    }

    #[test]
    fn try_from_jsonrpc_session_init_invalid_init_type() {
        let rpc = JsonRpcNotification::builder()
            .method("session.init".into())
            .add_param("session_id".into(), Value::from("s1"))
            .add_param("model".into(), Value::from("gpt-4"))
            .add_param("provider".into(), Value::from("openai"))
            .add_param("system".into(), Value::from("sys"))
            .add_param("init_type".into(), Value::from("invalid"))
            .add_param("timestamp".into(), {
                let tms = Utc::now();
                serde_json::to_value(tms).expect("Should convert to value")
            });
        let err = AgentEventAny::try_from(rpc).unwrap_err();
        assert!(matches!(err, AgentEventError::InvalidFieldType(_)));
    }

    #[test]
    fn try_from_jsonrpc_session_stop_ok() {
        let rpc = JsonRpcNotification::builder()
            .method("session.stop".into())
            .add_param("session_id".into(), Value::from("s1"))
            .add_param("success".into(), Value::from(true))
            .add_param("result".into(), Value::from(vec![json!({"text": "done"})]))
            .add_param("error".into(), Value::Null)
            .add_param("incomplete_tasks".into(), Value::Null)
            .add_param("timestamp".into(), {
                let tms = Utc::now();
                serde_json::to_value(tms).expect("Should convert to value")
            })
            .add_param("usage".into(), {
                let usg = Usage::default();
                serde_json::to_value(usg).expect("Should convert to value")
            });
        let any = AgentEventAny::try_from(rpc).unwrap();
        assert!(
            matches!(any, AgentEventAny::SessionStop(ref e) if e.success && e.result == Some(vec![AssistantMessagePart::Text(TextPart { text: "done".to_string() })]) && e.error.is_none() && e.usage.latency == 0)
        );
    }

    #[test]
    fn try_from_jsonrpc_user_prompt_submit_ok() {
        let rpc = JsonRpcNotification::builder()
            .method("user.prompt.submit".into())
            .add_param("session_id".into(), Value::from("s1"))
            .add_param("turn_id".into(), Value::from("t1"))
            .add_param("prompt".into(), Value::from("hello"))
            .add_param("timestamp".into(), {
                let tms = Utc::now();
                serde_json::to_value(tms).expect("Should convert to value")
            });
        let any = AgentEventAny::try_from(rpc).unwrap();
        assert!(matches!(any, AgentEventAny::UserPromptSubmit(ref e) if e.prompt == "hello"));
    }

    #[test]
    fn try_from_jsonrpc_stream_delta_no_default() {
        let rpc = JsonRpcNotification::builder()
            .method("stream.delta".into())
            .add_param("session_id".into(), Value::from("s1"))
            .add_param("turn_id".into(), Value::from("t1"))
            .add_param("delta".into(), Value::from("d"))
            .add_param("timestamp".into(), {
                let tms = Utc::now();
                serde_json::to_value(tms).expect("Should convert to value")
            });
        let any = AgentEventAny::try_from(rpc);
        assert!(any.is_err_and(
            |e| matches!(e, AgentEventError::MissingField(ref err) if err == "delta_type")
        ));
    }

    #[test]
    fn try_from_jsonrpc_stream_delta_thinking() {
        let rpc = JsonRpcNotification::builder()
            .method("stream.delta".into())
            .add_param("session_id".into(), Value::from("s1"))
            .add_param("turn_id".into(), Value::from("t1"))
            .add_param("delta".into(), Value::from("d"))
            .add_param(
                "delta_type".into(),
                serde_json::to_value(DeltaType::Thinking)
                    .expect("Should be able to convert to value"),
            )
            .add_param("timestamp".into(), {
                let tms = Utc::now();
                serde_json::to_value(tms).expect("Should convert to value")
            });
        let any = AgentEventAny::try_from(rpc).unwrap();
        assert!(
            matches!(any, AgentEventAny::StreamDelta(ref e) if matches!(e.delta_type, DeltaType::Thinking))
        );
    }

    #[test]
    fn try_from_jsonrpc_tool_call_ok() {
        let rpc = JsonRpcNotification::builder()
            .method("tool.call".into())
            .add_param("session_id".into(), Value::from("s1"))
            .add_param("turn_id".into(), Value::from("t1"))
            .add_param("name".into(), Value::from("read"))
            .add_param("input".into(), json!({"path": "/tmp"}))
            .add_param("timestamp".into(), {
                let tms = Utc::now();
                serde_json::to_value(tms).expect("Should convert to value")
            });
        let any = AgentEventAny::try_from(rpc).unwrap();
        assert!(matches!(any, AgentEventAny::ToolCall(ref e) if e.name == "read"));
    }

    #[test]
    fn try_from_jsonrpc_tool_result_ok() {
        let rpc = JsonRpcNotification::builder()
            .method("tool.result".into())
            .add_param("session_id".into(), Value::from("s1"))
            .add_param("turn_id".into(), Value::from("t1"))
            .add_param("tool_call_id".into(), Value::from("tc1"))
            .add_param("result".into(), json!({"Ok": "ok"}))
            .add_param("timestamp".into(), {
                let tms = Utc::now();
                serde_json::to_value(tms).expect("Should convert to value")
            });
        let any = AgentEventAny::try_from(rpc).unwrap();
        assert!(
            matches!(any, AgentEventAny::ToolResult(ref e) if matches!(e.result, ToolResult::Ok(ref s) if s == "ok"))
        );
    }

    #[test]
    fn try_from_jsonrpc_skill_load_ok() {
        let rpc = JsonRpcNotification::builder()
            .method("skill.load".into())
            .add_param("session_id".into(), Value::from("s1"))
            .add_param("turn_id".into(), Value::from("t1"))
            .add_param("skill_name".into(), Value::from("coding"))
            .add_param("timestamp".into(), {
                let tms = Utc::now();
                serde_json::to_value(tms).expect("Should convert to value")
            });
        let any = AgentEventAny::try_from(rpc).unwrap();
        assert!(matches!(any, AgentEventAny::SkillLoad(ref e) if e.skill_name == "coding"));
    }

    #[test]
    fn try_from_jsonrpc_assistant_response_ok() {
        let rpc = JsonRpcNotification::builder()
            .method("assistant.response".into())
            .add_param("session_id".into(), Value::from("s1"))
            .add_param("turn_id".into(), Value::from("t1"))
            .add_param(
                "content".into(),
                serde_json::to_value(vec![AssistantMessagePart::Text(TextPart {
                    text: "hi".to_string(),
                })])
                .expect("Should convert to value"),
            )
            .add_param("timestamp".into(), {
                let tms = Utc::now();
                serde_json::to_value(tms).expect("Should convert to value")
            });
        let any = AgentEventAny::try_from(rpc).unwrap();
        assert!(
            matches!(any, AgentEventAny::AssistantResponse(ref e) if e.content.len() == 1 && e.content.iter().all(|v| matches!(v, AssistantMessagePart::Text(_))))
        );
    }

    #[test]
    fn try_from_jsonrpc_assistant_response_with_tool_calls() {
        let rpc = JsonRpcNotification::builder()
            .method("assistant.response".into())
            .add_param("session_id".into(), Value::from("s1"))
            .add_param("turn_id".into(), Value::from("t1"))
            .add_param(
                "content".into(),
                serde_json::to_value(vec![AssistantMessagePart::ToolCall(ToolCallPart {
                    id: "tc1".to_string(),
                    name: "tool".to_string(),
                    arguments: "{}".to_string(),
                })])
                .expect("Should convert to value"),
            )
            .add_param("timestamp".into(), {
                let tms = Utc::now();
                serde_json::to_value(tms).expect("Should convert to value")
            });
        let any = AgentEventAny::try_from(rpc).unwrap();
        assert!(
            matches!(any, AgentEventAny::AssistantResponse(ref e) if e.content.len() == 1 && e.content.iter().all(|v| matches!(v, AssistantMessagePart::ToolCall(_))))
        );
    }

    #[test]
    fn try_from_jsonrpc_task() {
        let rpc = JsonRpcNotification::builder()
            .method("assistant.task".into())
            .add_param("session_id".into(), Value::from("s1"))
            .add_param("turn_id".into(), Value::from("t1"))
            .add_param("task_name".into(), Value::from("test"))
            .add_param(
                "task_status".into(),
                serde_json::to_value(TaskStatus::Done).unwrap(),
            )
            .add_param("timestamp".into(), {
                let tms = Utc::now();
                serde_json::to_value(tms).expect("Should convert to value")
            });
        let any = AgentEventAny::try_from(rpc).unwrap();
        assert!(
            matches!(any, AgentEventAny::Task(ref e) if e.task_name == "test" && e.task_status == TaskStatus::Done)
        );
    }

    #[test]
    fn try_from_jsonrpc_session_fork_ok() {
        let rpc = JsonRpcNotification::builder()
            .method("session.fork".into())
            .add_param("session_id".into(), Value::from("s1"))
            .add_param("parent_session_id".into(), Value::from("parent1"))
            .add_param("timestamp".into(), {
                let tms = Utc::now();
                serde_json::to_value(tms).expect("Should convert to value")
            });
        let any = AgentEventAny::try_from(rpc).unwrap();
        assert!(
            matches!(any, AgentEventAny::SessionFork(ref e) if e.session_id == "s1" && e.parent_session_id == "parent1")
        );
    }

    #[test]
    fn try_from_jsonrpc_unknown_method() {
        let rpc = JsonRpcNotification::builder()
            .method("unknown".into())
            .add_param("session_id".into(), Value::from("s1"))
            .add_param("timestamp".into(), {
                let tms = Utc::now();
                serde_json::to_value(tms).expect("Should convert to value")
            });
        let err = AgentEventAny::try_from(rpc).unwrap_err();
        assert!(matches!(err, AgentEventError::UnknownMethod(ref m) if m == "unknown"));
    }

    #[test]
    fn try_from_jsonrpc_missing_session_id() {
        let rpc = JsonRpcNotification::builder()
            .method("session.stop".into())
            .add_param("success".into(), Value::from(true))
            .add_param("timestamp".into(), {
                let tms = Utc::now();
                serde_json::to_value(tms).expect("Should convert to value")
            });
        let err = AgentEventAny::try_from(rpc).unwrap_err();
        assert!(matches!(err, AgentEventError::MissingField(ref m) if m == "session_id"));
    }
}
