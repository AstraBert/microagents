use futures_util::StreamExt;
use microagents_core::agent::MicroAgentBuilder;
use microagents_core::types::Agent;
use microagents_core::types::AgentError;
use microagents_core::types::ToolExecutionContext;
use microagents_core::types::ToolFunction;
use microagents_events::AgentEventAny;
use microagents_events::AssistantMessagePart;
use microagents_events::SessionInitType;
use microagents_events::types::ToolResult;
use serde_json::{Value, json};
use std::sync::Arc;

#[derive(Debug)]
struct WeatherTool;

#[async_trait::async_trait]
impl ToolFunction<()> for WeatherTool {
    fn name(&self) -> &'static str {
        "weather_tool"
    }

    fn description(&self) -> &'static str {
        "Get the weather for a give location"
    }

    fn input_schema(&self) -> Value {
        json!({
          "type": "object",
          "required": [
            "location"
          ],
          "properties": {
            "location": {
              "type": "string",
              "description": "Location to get the weather for"
            }
          }
        })
    }

    async fn execute(
        &self,
        input: Value,
        _ctx: &Arc<ToolExecutionContext<()>>,
    ) -> Result<ToolResult, AgentError> {
        let location = input["location"].as_str();
        if let Some(l) = location {
            return Ok(ToolResult::Ok(format!(
                "Weather in {l} is sunny with 27 degrees Celsius."
            )));
        }
        Ok(ToolResult::Err(
            "Could not find location in the tool input".into(),
        ))
    }
}

#[tokio::test]
async fn test_microagent_integration_openai() {
    match std::env::var("OPENAI_API_KEY") {
        Ok(_) => {}
        Err(_) => return,
    }
    let agent = MicroAgentBuilder::<()>::new(ToolExecutionContext::<()>::new(()))
        .model("gpt-5.4-mini")
        .provider("openai".to_string())
        .expect("Should load provider")
        .custom_instructions("Always call the weather_tool when asked about the weather")
        .add_tool(Arc::new(WeatherTool))
        .expect("Should be able to register tool")
        .build()
        .expect("Should be able to build the agent");
    let mut events = vec![];
    let mut stream = agent
        .run("What is the weather in San Francisco?".into(), None)
        .await
        .unwrap();
    while let Some(e) = stream.next().await {
        match e {
            Ok(ev) => events.push(ev),
            Err(err) => panic!("{}", err.to_string()),
        }
    }
    // session init + user prompt submit + tool call + assistant response + stop
    assert!(events.len() >= 5);
    let mut has_init = false;
    let mut has_user_prompt_submit = false;
    let mut has_assistant_response = false;
    let mut has_tool_call = false;
    let mut has_tool_result = false;
    let mut has_stop = false;
    let mut has_stream = false;
    let mut has_skill = false; // this should be false
    for ev in events {
        match ev {
            AgentEventAny::SessionInit(e) => {
                has_init = true;
                assert_eq!(e.init_type, SessionInitType::Start);
                assert_eq!(e.provider, "openai");
                assert_eq!(e.model, "gpt-5.4-mini");
            }
            AgentEventAny::AssistantResponse(_) => has_assistant_response = true,
            AgentEventAny::StreamDelta(_) => has_stream = true,
            AgentEventAny::ToolCall(_) => has_tool_call = true,
            AgentEventAny::ToolResult(e) => {
                has_tool_result = true;
                assert!(
                    matches!(e.result, ToolResult::Ok(ref res) if res == "Weather in San Francisco is sunny with 27 degrees Celsius.")
                )
            }
            AgentEventAny::UserPromptSubmit(e) => {
                has_user_prompt_submit = true;
                assert_eq!(e.prompt, "What is the weather in San Francisco?");
            }
            AgentEventAny::SessionStop(e) => {
                has_stop = true;
                assert!(e.result.is_some());
                assert!(e.error.is_none());
            }
            AgentEventAny::SkillLoad(_) => has_skill = true,
            _ => unreachable!("AgentEventAny should not reach this branch"),
        }
    }
    assert!(has_init);
    assert!(has_user_prompt_submit);
    assert!(has_tool_call);
    assert!(has_tool_result);
    assert!(has_stream);
    assert!(has_assistant_response);
    assert!(has_stop);
    assert!(!has_skill);
}

#[tokio::test]
async fn test_microagent_integration_anthropic() {
    match std::env::var("ANTHROPIC_API_KEY") {
        Ok(_) => {}
        Err(_) => return,
    }
    let agent = MicroAgentBuilder::<()>::new(ToolExecutionContext::<()>::new(()))
        .model("claude-sonnet-5")
        .provider("anthropic".to_string())
        .expect("Should load provider")
        .custom_instructions("Always call the weather_tool when asked about the weather")
        .add_tool(Arc::new(WeatherTool))
        .expect("Should be able to register tool")
        .build()
        .expect("Should be able to build the agent");
    let mut events = vec![];
    let mut stream = agent
        .run("What is the weather in San Francisco?".into(), None)
        .await
        .unwrap();
    while let Some(e) = stream.next().await {
        match e {
            Ok(ev) => events.push(ev),
            Err(err) => panic!("{}", err.to_string()),
        }
    }
    // session init + user prompt submit + tool call + assistant response + stop
    assert!(events.len() >= 5);
    let mut has_init = false;
    let mut has_user_prompt_submit = false;
    let mut has_assistant_response = false;
    let mut has_tool_call = false;
    let mut has_tool_result = false;
    let mut has_stop = false;
    let mut has_stream = false;
    let mut has_skill = false; // this should be false
    for ev in events {
        match ev {
            AgentEventAny::SessionInit(e) => {
                has_init = true;
                assert_eq!(e.init_type, SessionInitType::Start);
                assert_eq!(e.provider, "anthropic");
                assert_eq!(e.model, "claude-sonnet-5");
            }
            AgentEventAny::AssistantResponse(ar) => {
                has_assistant_response = true;
                assert!(ar.content.iter().any(
                    |t| matches!(t, AssistantMessagePart::ToolCall(tc) if tc.name == "weather_tool")
                ))
            }
            AgentEventAny::StreamDelta(_) => has_stream = true,
            AgentEventAny::ToolCall(_) => has_tool_call = true,
            AgentEventAny::ToolResult(e) => {
                has_tool_result = true;
                assert!(
                    matches!(e.result, ToolResult::Ok(ref res) if res == "Weather in San Francisco is sunny with 27 degrees Celsius.")
                )
            }
            AgentEventAny::UserPromptSubmit(e) => {
                has_user_prompt_submit = true;
                assert_eq!(e.prompt, "What is the weather in San Francisco?");
            }
            AgentEventAny::SessionStop(e) => {
                has_stop = true;
                assert!(e.result.is_some());
                assert!(e.error.is_none());
            }
            AgentEventAny::SkillLoad(_) => has_skill = true,
            _ => unreachable!("AgentEventAny should not reach this branch"),
        }
    }
    assert!(has_init);
    assert!(has_user_prompt_submit);
    assert!(has_tool_call);
    assert!(has_tool_result);
    assert!(has_stream);
    assert!(has_assistant_response);
    assert!(has_stop);
    assert!(!has_skill);
}
