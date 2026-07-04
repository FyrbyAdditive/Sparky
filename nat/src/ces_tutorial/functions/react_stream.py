"""Streaming ReAct agent: speaks the Final Answer as it decodes.

The stock NAT react_agent runs its LangGraph to completion and returns one
buffered string, so every tool-using turn was silent until the ENTIRE
trace (thoughts, tool calls, final decode of up to 2048 tokens) finished —
the single biggest felt-latency cost in the system (2-6s of dead air).

This vendored variant (config subclasses the stock one, so all knobs carry
over) builds the identical ReActAgentGraph but consumes it through
``astream_events``: LLM tokens stream out of the graph as they decode, and
once a generation's accumulated text contains ``Final Answer:`` the tail
is forwarded token-by-token. The voice pipeline starts speaking on the
first forwarded sentence instead of after the whole trace.

Guardrails:
- Tokens before the marker (thoughts/actions) are never forwarded.
- If the stream ends without a marker (parse failure path), fall back to
  the buffered last-message content run through the same scaffolding
  cleaner the non-streaming path used — behavior-identical to before.
- Known trade-off: if a generation emits ``Final Answer:`` and the ReAct
  parser still rejects it (malformed trace, rare), a partial answer may be
  spoken before the retry's answer. Accepted; the alternative is buffering
  everything again.

Vendored from nat.agent.react_agent.register (Apache-2.0).
"""

import logging
from collections.abc import AsyncGenerator

from nat.agent.react_agent.register import ReActAgentWorkflowConfig
from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.api_server import ChatRequest, ChatRequestOrMessage
from nat.utils.type_converter import GlobalTypeConverter

logger = logging.getLogger(__name__)

FINAL_MARKER = "Final Answer:"


class StreamingReActAgentConfig(ReActAgentWorkflowConfig, name="react_agent_streaming"):
    """Stock react_agent config, streaming execution."""


def _clean_buffered_reply(text: str) -> str:
    """Fallback sanitizer for the non-streamed path (mirrors router_agent)."""
    from ces_tutorial.functions.router_agent import _clean_agent_reply
    return _clean_agent_reply(text)


@register_function(config_type=StreamingReActAgentConfig,
                   framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def streaming_react_agent_workflow(config: StreamingReActAgentConfig,
                                         builder: Builder):
    from langchain_core.messages import trim_messages
    from langgraph.graph.state import CompiledStateGraph

    from nat.agent.react_agent.agent import (ReActAgentGraph, ReActGraphState,
                                             create_react_agent_prompt)

    prompt = create_react_agent_prompt(config)
    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    tools = await builder.get_tools(tool_names=config.tool_names,
                                    wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    if not tools:
        raise ValueError(f"No tools specified for streaming ReAct Agent '{config.llm_name}'")

    graph: CompiledStateGraph = await ReActAgentGraph(
        llm=llm,
        prompt=prompt,
        tools=tools,
        use_tool_schema=config.include_tool_input_schema_in_tool_description,
        detailed_logs=config.verbose,
        log_response_max_chars=config.log_response_max_chars,
        retry_agent_response_parsing_errors=config.retry_agent_response_parsing_errors,
        parse_agent_response_max_retries=config.parse_agent_response_max_retries,
        tool_call_max_retries=config.tool_call_max_retries,
        pass_tool_call_errors_to_agent=config.pass_tool_call_errors_to_agent,
        normalize_tool_input_quotes=config.normalize_tool_input_quotes).build_graph()

    async def _stream_fn(chat_request_or_message: ChatRequestOrMessage) -> AsyncGenerator[str]:
        message = GlobalTypeConverter.get().convert(chat_request_or_message,
                                                    to_type=ChatRequest)
        messages = trim_messages(messages=[m.model_dump() for m in message.messages],
                                 max_tokens=config.max_history,
                                 strategy="last",
                                 token_counter=len,
                                 start_on="human",
                                 include_system=True)
        state = ReActGraphState(messages=messages)

        forwarded = False
        gen_buffers: dict[str, str] = {}
        final_output = None
        async for ev in graph.astream_events(
                state, config={"recursion_limit": (config.max_tool_calls + 1) * 2},
                version="v2"):
            kind = ev.get("event")
            if kind == "on_chat_model_stream":
                chunk = ev.get("data", {}).get("chunk")
                content = getattr(chunk, "content", None) or ""
                if not content:
                    continue
                if forwarded:
                    yield content
                    continue
                run_id = str(ev.get("run_id"))
                buf = gen_buffers.get(run_id, "") + content
                gen_buffers[run_id] = buf
                idx = buf.find(FINAL_MARKER)
                if idx >= 0:
                    tail = buf[idx + len(FINAL_MARKER):].lstrip()
                    if tail:
                        forwarded = True
                        yield tail
            elif kind == "on_chain_end" and ev.get("name") == "LangGraph":
                final_output = ev.get("data", {}).get("output")

        if not forwarded:
            # parse-retry / non-streaming-model fallback: identical to the
            # old buffered behavior
            content = ""
            try:
                out_state = ReActGraphState(**final_output) if isinstance(
                    final_output, dict) else final_output
                content = str(out_state.messages[-1].content)
            except Exception as e:
                logger.error(f"streaming react agent: no final output ({e})")
            if content:
                yield _clean_buffered_reply(content)

    yield FunctionInfo.create(stream_fn=_stream_fn, description=config.description)
