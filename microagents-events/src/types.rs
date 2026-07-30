use std::fmt::Debug;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use thiserror::Error;

const JSONRPC: &str = "2.0";

/// Errors that can occur when parsing a [`JsonRpcNotification`] into an [`AgentEventAny`].
#[derive(Debug, Clone, Error)]
#[non_exhaustive]
pub enum AgentEventError {
    /// A required field was missing from the JSON-RPC params.
    #[error("Missing required field: {0}")]
    MissingField(String),
    /// A field had an unexpected type.
    #[error("Invalid type for field: {0}")]
    InvalidFieldType(String),
    /// The JSON-RPC method name is not recognized.
    #[error("Unknown method: {0}")]
    UnknownMethod(String),
}

/// Result of a tool execution, either success with output or failure with an error message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ToolResult {
    /// Tool executed successfully.
    Ok(String),
    /// Tool execution failed.
    Err(String),
}

/// A JSON-RPC 2.0 notification message.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct JsonRpcNotification {
    pub jsonrpc: String,
    pub method: String,
    pub params: Map<String, Value>,
}

impl JsonRpcNotification {
    pub fn builder() -> Self {
        Self {
            jsonrpc: JSONRPC.into(),
            method: String::new(),
            params: Map::new(),
        }
    }

    pub fn method(mut self, method: String) -> Self {
        self.method = method;
        self
    }

    pub fn add_param(mut self, key: String, value: Value) -> Self {
        self.params.insert(key, value);
        self
    }
}

/// Trait for events that can be converted to JSON-RPC notifications and carry a session ID.
pub trait AgentEvent: Debug + Send + Sync {
    /// Convert this event into a [`JsonRpcNotification`].
    fn to_jsonrpc(&self) -> JsonRpcNotification;
    /// Return the session ID associated with this event.
    fn session_id(&self) -> String;
}

#[cfg(test)]
mod tests {

    use serde_json::json;

    use super::*;

    #[test]
    fn test_value_from_tool_result() {
        let trs = ToolResult::Ok("success!".to_string());
        let trf = ToolResult::Err("error!".to_string());
        let value_s = serde_json::to_value(trs).expect("Should be able to convert to value");
        let value_f = serde_json::to_value(trf).expect("Should be able to convert to value");
        assert_eq!(
            value_s,
            json!({
                "Ok": "success!",
            })
        );
        assert_eq!(
            value_f,
            json!({
                "Err": "error!",
            })
        );
    }

    #[test]
    fn test_jsonrpc_notification_builder() {
        let jsonrpc = JsonRpcNotification::builder();
        assert_eq!(jsonrpc.jsonrpc, JSONRPC.to_string());
        assert_eq!(jsonrpc.method, String::new());
        assert_eq!(jsonrpc.params, Map::<String, Value>::new());

        let j = jsonrpc
            .method("test".into())
            .add_param("test".into(), Value::from("string"))
            .add_param("number".into(), Value::from(1));

        assert_eq!(j.method, "test".to_string());
        assert_eq!(j.params.len(), 2);
        assert!(j.params.get("test").is_some_and(|v| v.is_string()));
        assert!(j.params.get("number").is_some_and(|v| v.is_number()));
    }
}
