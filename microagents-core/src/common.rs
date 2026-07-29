use std::{
    collections::{HashMap, HashSet},
    fs,
    sync::Arc,
};

use chrono::{DateTime, Utc};
use llms_sdk::{
    Message, MessagePart, MessageRole, TextPart, ThinkingPart, ToolCallPart, ToolResultPart,
};
use microagents_events::{AgentEventAny, AssistantMessagePart, TaskStatus, types::ToolResult};
use serde_json::Value;

use crate::{
    agent::MicroAgentBuilderError,
    types::{AgentError, ToolExecutionContext, ToolFunction},
};

const AGENTS_MD: &str = "AGENTS.md";

/// Verify that an environment variable is set.
///
/// Returns `Ok(())` if the variable exists, otherwise propagates the [`VarError`].
pub fn check_env_var(api_key: &str) -> Result<String, std::env::VarError> {
    let v = std::env::var(api_key)?;
    Ok(v)
}

/// Convert a persisted [`AgentEventAny`] back into an SDK [`Message`].
///
/// Only events that correspond to chat roles (`User`, `Assistant`, `Tool`)
/// produce a message. All other variants return [`None`].
pub fn convert_event_to_message(event: AgentEventAny) -> Option<Message> {
    match event {
        AgentEventAny::UserPromptSubmit(p) => Some(Message {
            role: MessageRole::User,
            content: vec![MessagePart::Text(TextPart::new(p.prompt))],
        }),
        AgentEventAny::AssistantResponse(p) => {
            let mut content: Vec<MessagePart> = vec![];
            for c in p.content {
                match c {
                    AssistantMessagePart::Text(t) => {
                        content.push(MessagePart::Text(TextPart { text: t.text }))
                    }
                    AssistantMessagePart::Thinking(t) => {
                        content.push(MessagePart::Thinking(ThinkingPart {
                            thinking: t.thinking,
                            signature: t.signature,
                        }))
                    }
                    AssistantMessagePart::ToolCall(tc) => {
                        content.push(MessagePart::ToolCall(ToolCallPart {
                            id: tc.id,
                            name: tc.name,
                            arguments: tc.arguments,
                        }));
                    }
                }
            }
            Some(Message {
                role: MessageRole::Assistant,
                content,
            })
        }
        AgentEventAny::ToolResult(p) => {
            let result = match p.result {
                ToolResult::Ok(r) => format!("Tool call succeeded: {}", r),
                ToolResult::Err(r) => format!("Tool call failed: {}", r),
                _ => unreachable!("ToolResult should not reach this branch"),
            };
            Some(Message {
                role: MessageRole::Tool,
                content: vec![MessagePart::ToolResult(ToolResultPart {
                    tool_call_id: p.tool_call_id,
                    result,
                })],
            })
        }
        _ => None,
    }
}

/// Get incomplete tasks from the events of a previous session, comparing them with the
/// reported incomplete tasks to ensure no data loss/discrepancies
pub fn get_incomplete_tasks(
    events: &[AgentEventAny],
) -> Result<HashMap<String, TaskStatus>, AgentError> {
    let (tasks_hashmap, incomplete_tasks, last_task_ts, last_stop_ts) = collect_events(events);

    let incomplete_by_status: HashMap<String, TaskStatus> = tasks_hashmap
        .into_iter()
        .filter(|(_, s)| *s != TaskStatus::Done)
        .collect();

    // Use >= instead of strict > to allow for
    // some tiny deltas not being captured by the
    // Utc time
    let stop_is_authoritative = matches!(
        (last_stop_ts, last_task_ts),
        (Some(stop), Some(task)) if stop >= task
    );

    if stop_is_authoritative && let Some(incomplete) = incomplete_tasks.filter(|v| !v.is_empty()) {
        return validate_incomplete_tasks(&incomplete_by_status, &incomplete);
    }

    Ok(incomplete_by_status)
}

#[allow(clippy::type_complexity)]
fn collect_events(
    events: &[AgentEventAny],
) -> (
    HashMap<String, TaskStatus>,
    Option<Vec<String>>,
    Option<DateTime<Utc>>,
    Option<DateTime<Utc>>,
) {
    let mut tasks_hashmap: HashMap<String, TaskStatus> = HashMap::new();
    let mut incomplete_tasks: Option<Vec<String>> = None;
    let mut last_task_ts: Option<DateTime<Utc>> = None;
    let mut last_stop_ts: Option<DateTime<Utc>> = None;

    for ev in events {
        match ev {
            AgentEventAny::Task(t) => {
                tasks_hashmap.insert(t.task_name.clone(), t.task_status);
                last_task_ts = Some(t.timestamp);
            }
            AgentEventAny::SessionStop(s) => {
                incomplete_tasks = s.incomplete_tasks.clone();
                last_stop_ts = Some(s.timestamp);
            }
            _ => {}
        }
    }

    (tasks_hashmap, incomplete_tasks, last_task_ts, last_stop_ts)
}

fn validate_incomplete_tasks(
    by_status: &HashMap<String, TaskStatus>,
    reported: &[String],
) -> Result<HashMap<String, TaskStatus>, AgentError> {
    if by_status.is_empty() {
        return Err(AgentError::TaskLoadingError(
            "Incomplete tasks were reported but none could be inferred from events".to_string(),
        ));
    }

    let inferred: HashSet<&String> = by_status.keys().collect();
    let reported: HashSet<&String> = reported.iter().collect();
    if inferred != reported {
        return Err(AgentError::TaskLoadingError(
            "Reported incomplete tasks differ from those inferred from events".to_string(),
        ));
    }

    Ok(by_status.clone())
}

/// Validate tool arguments against its JSON schema and then execute it.
///
/// This is the canonical entry-point for invoking a [`ToolFunction`] from the
/// agent runtime. It first checks schema conformance with `jsonschema`, then
/// calls [`ToolFunction::execute`].
pub async fn call_tool<Ctx: Send + Sync + 'static>(
    tool: Arc<dyn ToolFunction<Ctx>>,
    tool_args: Value,
    tool_context: Arc<ToolExecutionContext<Ctx>>,
) -> Result<ToolResult, AgentError> {
    jsonschema::validate(&tool.input_schema(), &tool_args)
        .map_err(|e| AgentError::ToolCallError(e.to_string()))?;
    let result = tool.execute(tool_args, &tool_context).await?;
    Ok(result)
}

/// Load the content of an AGENTS.md file in the current directory, if it exists
pub fn load_agents_md() -> Result<Option<String>, MicroAgentBuilderError> {
    match fs::read_to_string(AGENTS_MD) {
        Ok(s) => Ok(Some(s)),
        Err(e) => match e.kind() {
            std::io::ErrorKind::NotFound => Ok(None),
            _ => Err(MicroAgentBuilderError::AgentsMdResolutionError(e)),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use microagents_events::{
        AssistantResponseEvent, SessionInitEvent, SessionInitType, SessionStopEvent,
        SkillLoadEvent, StreamDeltaEvent, TaskEvent, TaskStatus, TextPart as AssistantTextPart,
        ToolCallEvent, ToolCallPart as AssistantToolCallPart, ToolResultEvent, Usage,
        UserPromptSubmitEvent,
    };

    #[test]
    fn test_convert_user_prompt_submit() {
        let event = AgentEventAny::UserPromptSubmit(UserPromptSubmitEvent {
            session_id: "s1".into(),
            turn_id: "t1".into(),
            prompt: "hello".into(),
            timestamp: Utc::now(),
        });
        let msg = convert_event_to_message(event).unwrap();
        assert_eq!(msg.role, MessageRole::User);
        assert!(matches!(msg.content[0], MessagePart::Text(ref t) if t.text == "hello"));
    }

    #[test]
    fn test_convert_assistant_response_without_tool_calls() {
        let event = AgentEventAny::AssistantResponse(AssistantResponseEvent {
            session_id: "s1".into(),
            turn_id: "t1".into(),
            content: vec![AssistantMessagePart::Text(AssistantTextPart {
                text: "hi there".to_string(),
            })],
            timestamp: Utc::now(),
        });
        let msg = convert_event_to_message(event).unwrap();
        assert_eq!(msg.role, MessageRole::Assistant);
        assert!(matches!(msg.content[0], MessagePart::Text(ref t) if t.text == "hi there"));
    }

    #[test]
    fn test_convert_assistant_response_with_tool_calls() {
        let event = AgentEventAny::AssistantResponse(AssistantResponseEvent {
            session_id: "s1".into(),
            turn_id: "t1".into(),
            content: vec![
                AssistantMessagePart::Text(AssistantTextPart {
                    text: "calling tool".to_string(),
                }),
                AssistantMessagePart::ToolCall(AssistantToolCallPart {
                    id: "tc1".to_string(),
                    name: "my_tool".to_string(),
                    arguments: "{\"x\":1}".into(),
                }),
            ],
            timestamp: Utc::now(),
        });
        let msg = convert_event_to_message(event).unwrap();
        assert_eq!(msg.role, MessageRole::Assistant);
        let calls: Vec<MessagePart> = msg
            .content
            .iter()
            .filter(|&t| matches!(t, MessagePart::ToolCall(_)))
            .cloned()
            .collect();
        assert_eq!(calls.len(), 1);
        if let MessagePart::ToolCall(tc) = calls[0].clone() {
            assert_eq!(tc.id, "tc1");
            assert_eq!(tc.name, "my_tool");
            assert_eq!(tc.arguments, "{\"x\":1}");
        }
    }

    #[test]
    fn test_convert_tool_result_ok() {
        let event = AgentEventAny::ToolResult(ToolResultEvent {
            session_id: "s1".into(),
            turn_id: "t1".into(),
            result: ToolResult::Ok("done".into()),
            tool_call_id: "tc1".into(),
            timestamp: Utc::now(),
        });
        let msg = convert_event_to_message(event).unwrap();
        assert_eq!(msg.role, MessageRole::Tool);
        let first = msg.content.first().unwrap();
        if let MessagePart::ToolResult(tr) = first {
            assert_eq!(tr.result, "Tool call succeeded: done");
            assert_eq!(tr.tool_call_id, "tc1");
        }
    }

    #[test]
    fn test_convert_tool_result_err() {
        let event = AgentEventAny::ToolResult(ToolResultEvent {
            session_id: "s1".into(),
            turn_id: "t1".into(),
            result: ToolResult::Err("oops".into()),
            tool_call_id: "tc2".into(),
            timestamp: Utc::now(),
        });
        let msg = convert_event_to_message(event).unwrap();
        assert_eq!(msg.role, MessageRole::Tool);
        let first = msg.content.first().unwrap();
        if let MessagePart::ToolResult(tr) = first {
            assert_eq!(tr.result, "Tool call failed: oops");
            assert_eq!(tr.tool_call_id, "tc2");
        }
    }

    #[test]
    fn test_convert_other_events_return_none() {
        assert!(
            convert_event_to_message(AgentEventAny::SessionInit(SessionInitEvent {
                session_id: "s1".into(),
                model: "m".into(),
                provider: "p".into(),
                system: "sys".into(),
                init_type: SessionInitType::Start,
                timestamp: Utc::now(),
            }))
            .is_none()
        );

        assert!(
            convert_event_to_message(AgentEventAny::SessionStop(SessionStopEvent {
                session_id: "s1".into(),
                success: true,
                result: None,
                error: None,
                timestamp: Utc::now(),
                usage: Usage::default(),
                incomplete_tasks: None,
            }))
            .is_none()
        );

        assert!(
            convert_event_to_message(AgentEventAny::StreamDelta(StreamDeltaEvent {
                session_id: "s1".into(),
                turn_id: "t1".into(),
                delta: "d".into(),
                delta_type: microagents_events::DeltaType::Text,
                timestamp: Utc::now(),
            }))
            .is_none()
        );

        assert!(
            convert_event_to_message(AgentEventAny::ToolCall(ToolCallEvent {
                session_id: "s1".into(),
                turn_id: "t1".into(),
                name: "tool".into(),
                input: Value::Null,
                timestamp: Utc::now(),
            }))
            .is_none()
        );

        assert!(
            convert_event_to_message(AgentEventAny::SkillLoad(SkillLoadEvent {
                session_id: "s1".into(),
                turn_id: "t1".into(),
                skill_name: "skill".into(),
                timestamp: Utc::now(),
            }))
            .is_none()
        );
    }

    #[derive(Debug)]
    struct DummyTool {
        schema: Value,
    }

    #[async_trait::async_trait]
    impl ToolFunction<()> for DummyTool {
        fn name(&self) -> &'static str {
            "dummy"
        }
        fn description(&self) -> &'static str {
            "desc"
        }
        fn input_schema(&self) -> Value {
            self.schema.clone()
        }
        async fn execute(
            &self,
            _input: Value,
            _ctx: &Arc<ToolExecutionContext<()>>,
        ) -> Result<ToolResult, AgentError> {
            Ok(ToolResult::Ok("ok".into()))
        }
    }

    #[tokio::test]
    async fn test_call_tool_validates_and_executes() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" }
            },
            "required": ["name"]
        });
        let tool = Arc::new(DummyTool { schema });
        let ctx = Arc::new(ToolExecutionContext::new(()));
        let args = serde_json::json!({"name": "world"});
        let result = call_tool(tool, args, ctx).await.unwrap();
        assert!(matches!(result, ToolResult::Ok(ref s) if s == "ok"));
    }

    #[tokio::test]
    async fn test_call_tool_schema_validation_fails() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "count": { "type": "integer" }
            },
            "required": ["count"]
        });
        let tool = Arc::new(DummyTool { schema });
        let ctx = Arc::new(ToolExecutionContext::new(()));
        let args = serde_json::json!({"count": "not a number"});
        let err = call_tool(tool, args, ctx).await.unwrap_err();
        match err {
            AgentError::ToolCallError(_) => {}
            other => panic!("expected ToolCallError, got {:?}", other),
        }
    }

    #[test]
    fn test_get_incomplete_tasks_no_session_stop_returns_inferred() {
        // No SessionStop event: tasks inferred from events only (Done filtered out).
        let events = vec![
            AgentEventAny::Task(TaskEvent {
                session_id: "s1".into(),
                turn_id: "t1".into(),
                task_name: "task_a".into(),
                task_status: TaskStatus::InProgress,
                timestamp: Utc::now(),
            }),
            AgentEventAny::Task(TaskEvent {
                session_id: "s1".into(),
                turn_id: "t1".into(),
                task_name: "task_b".into(),
                task_status: TaskStatus::Done,
                timestamp: Utc::now(),
            }),
            AgentEventAny::Task(TaskEvent {
                session_id: "s1".into(),
                turn_id: "t1".into(),
                task_name: "task_c".into(),
                task_status: TaskStatus::Queued,
                timestamp: Utc::now(),
            }),
        ];
        let result = get_incomplete_tasks(&events).expect("Should infer incomplete tasks");
        assert_eq!(result.len(), 2);
        assert_eq!(result.get("task_a"), Some(&TaskStatus::InProgress));
        assert_eq!(result.get("task_c"), Some(&TaskStatus::Queued));
        assert!(!result.contains_key("task_b"));
    }

    #[test]
    fn test_get_incomplete_tasks_no_tasks_returns_empty() {
        // No Task events and no incomplete tasks reported: empty result.
        let events: Vec<AgentEventAny> = vec![];
        let result = get_incomplete_tasks(&events).expect("Should return empty map");
        assert!(result.is_empty());
    }

    #[test]
    fn test_get_incomplete_tasks_session_stop_reports_matching() {
        // SessionStop reports incomplete tasks that match the inferred ones.
        let events = vec![
            AgentEventAny::Task(TaskEvent {
                session_id: "s1".into(),
                turn_id: "t1".into(),
                task_name: "task_a".into(),
                task_status: TaskStatus::InProgress,
                timestamp: Utc::now(),
            }),
            AgentEventAny::Task(TaskEvent {
                session_id: "s1".into(),
                turn_id: "t1".into(),
                task_name: "task_b".into(),
                task_status: TaskStatus::Queued,
                timestamp: Utc::now(),
            }),
            AgentEventAny::SessionStop(SessionStopEvent {
                session_id: "s1".into(),
                success: false,
                result: None,
                error: None,
                timestamp: Utc::now(),
                usage: Usage::default(),
                incomplete_tasks: Some(vec!["task_a".into(), "task_b".into()]),
            }),
        ];
        let result = get_incomplete_tasks(&events).expect("Should return matching tasks");
        assert_eq!(result.len(), 2);
        assert_eq!(result.get("task_a"), Some(&TaskStatus::InProgress));
        assert_eq!(result.get("task_b"), Some(&TaskStatus::Queued));
    }

    #[test]
    fn test_get_incomplete_tasks_session_stop_empty_uses_inferred() {
        // SessionStop reports empty incomplete_tasks: fall back to inferred tasks.
        let events = vec![
            AgentEventAny::Task(TaskEvent {
                session_id: "s1".into(),
                turn_id: "t1".into(),
                task_name: "task_a".into(),
                task_status: TaskStatus::InProgress,
                timestamp: Utc::now(),
            }),
            AgentEventAny::SessionStop(SessionStopEvent {
                session_id: "s1".into(),
                success: true,
                result: None,
                error: None,
                timestamp: Utc::now(),
                usage: Usage::default(),
                incomplete_tasks: Some(vec![]),
            }),
        ];
        let result = get_incomplete_tasks(&events).expect("Should fall back to inferred tasks");
        assert_eq!(result.len(), 1);
        assert_eq!(result.get("task_a"), Some(&TaskStatus::InProgress));
    }

    #[test]
    fn test_get_incomplete_tasks_session_stop_none_uses_inferred() {
        // SessionStop with incomplete_tasks=None: fall back to inferred tasks.
        let events = vec![
            AgentEventAny::Task(TaskEvent {
                session_id: "s1".into(),
                turn_id: "t1".into(),
                task_name: "task_a".into(),
                task_status: TaskStatus::Queued,
                timestamp: Utc::now(),
            }),
            AgentEventAny::SessionStop(SessionStopEvent {
                session_id: "s1".into(),
                success: true,
                result: None,
                error: None,
                timestamp: Utc::now(),
                usage: Usage::default(),
                incomplete_tasks: None,
            }),
        ];
        let result = get_incomplete_tasks(&events).expect("Should fall back to inferred tasks");
        assert_eq!(result.len(), 1);
        assert_eq!(result.get("task_a"), Some(&TaskStatus::Queued));
    }

    #[test]
    fn test_get_incomplete_tasks_mismatch_errors() {
        // SessionStop reports incomplete tasks that differ from inferred: error.
        let events = vec![
            AgentEventAny::Task(TaskEvent {
                session_id: "s1".into(),
                turn_id: "t1".into(),
                task_name: "task_a".into(),
                task_status: TaskStatus::InProgress,
                timestamp: Utc::now(),
            }),
            AgentEventAny::SessionStop(SessionStopEvent {
                session_id: "s1".into(),
                success: false,
                result: None,
                error: None,
                timestamp: Utc::now(),
                usage: Usage::default(),
                incomplete_tasks: Some(vec!["task_a".into(), "task_x".into()]),
            }),
        ];
        let err = get_incomplete_tasks(&events).expect_err("Should fail on mismatch");
        assert!(matches!(err, AgentError::TaskLoadingError(_)));
    }

    #[test]
    fn test_get_incomplete_tasks_done_task_filtered_out() {
        // A task marked Done should not appear even if reported as incomplete.
        let events = vec![
            AgentEventAny::Task(TaskEvent {
                session_id: "s1".into(),
                turn_id: "t1".into(),
                task_name: "task_a".into(),
                task_status: TaskStatus::Done,
                timestamp: Utc::now(),
            }),
            AgentEventAny::SessionStop(SessionStopEvent {
                session_id: "s1".into(),
                success: true,
                result: None,
                error: None,
                timestamp: Utc::now(),
                usage: Usage::default(),
                incomplete_tasks: Some(vec!["task_a".into()]),
            }),
        ];
        let err =
            get_incomplete_tasks(&events).expect_err("Should fail on mismatch with Done task");
        assert!(matches!(err, AgentError::TaskLoadingError(_)));
    }

    #[test]
    fn test_load_agents_md() {
        let instr = "you are a helpful assistant";
        fs::write(AGENTS_MD, instr).expect("Should be able to write file");
        let loaded = load_agents_md().expect("Should be able to load AGENTS.md content");
        assert_eq!(loaded, Some(instr.to_string()));
    }
}
