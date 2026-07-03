import logging
import re

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.component_ref import LLMRef, FunctionRef

logger = logging.getLogger(__name__)

_REACT_NOISE = re.compile(
    r"^(?:Thought|Action|Action Input|Observation)\s*:.*$", re.MULTILINE)
_REACT_PARSE_ERR = re.compile(
    r"Parsing LLM output produced both a final answer and a parse-able action::?")


def _clean_agent_reply(content: str) -> str:
    """Everything the agent returns is SPOKEN — a ReAct parse hiccup once
    put 'Thought:/Action:/Parsing LLM output...' through the robot's voice.
    Keep only the part after the last Final Answer and drop scaffolding."""
    if not content:
        return content
    if "Final Answer:" in content:
        content = content.rsplit("Final Answer:", 1)[1]
    content = _REACT_PARSE_ERR.sub("", content)
    content = _REACT_NOISE.sub("", content)
    content = content.strip()
    return content or "Sorry, I lost my train of thought there. Could you ask again?"


class RouterAgentConfig(FunctionBaseConfig, name="ces_tutorial_router_agent"):
    """A workflow that routes requests between chitchat, image understanding, and a full agent."""
    
    router: FunctionRef = Field(
        description="The router function to determine intent"
    )
    chitchat_llm: LLMRef = Field(
        description="The LLM to use for chitchat responses"
    )
    image_llm: LLMRef = Field(
        description="The LLM to use for image understanding"
    )
    agent: FunctionRef = Field(
        description="The agent function to handle complex requests"
    )


@register_function(config_type=RouterAgentConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def router_agent_fn(config: RouterAgentConfig, builder: Builder):
    """Route between chitchat LLM, image LLM, and agent based on user intent."""
    
    from ces_tutorial.openai_chat_request import OpenAIChatRequest as ChatRequest
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    
    # Get the router function
    router_function = await builder.get_function(name=config.router)
    
    # Get the chitchat LLM
    chitchat_llm = await builder.get_llm(llm_name=config.chitchat_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    
    # Get the image LLM
    image_llm = await builder.get_llm(llm_name=config.image_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    
    # Get the agent function
    agent_function = await builder.get_function(name=config.agent)
    
    def _redact_images_from_content(content):
        """Extract only text from multimodal content."""
        # Check if content is iterable (list, ValidatorIterator, etc.) but not a string
        if not isinstance(content, str) and hasattr(content, '__iter__'):
            # Convert to list first to handle ValidatorIterator
            content_list = list(content) if not isinstance(content, list) else content
            text_parts = [item.get('text', '') for item in content_list if isinstance(item, dict) and item.get('type') == 'text']
            return ' '.join(text_parts) if text_parts else ''
        return content
    
    def _convert_to_langchain_messages(messages, redact_images=False):
        """Convert OpenAI format messages to LangChain messages.

        All system messages are merged into a single one at the front —
        strict chat templates (e.g. Qwen) 400 on system messages anywhere
        but position 0.
        """
        system_parts = []
        conversation = []
        for msg in messages:
            msg_dict = msg.model_dump() if hasattr(msg, 'model_dump') else dict(msg)
            role = msg_dict.get('role')
            content = msg_dict.get('content')

            # Optionally redact images
            if redact_images:
                content = _redact_images_from_content(content)

            if role == 'system':
                if isinstance(content, str) and content:
                    system_parts.append(content)
            elif role == 'user':
                conversation.append(HumanMessage(content=content))
            elif role == 'assistant':
                conversation.append(AIMessage(content=content))

        langchain_messages = []
        if system_parts:
            langchain_messages.append(SystemMessage(content=" ".join(system_parts)))
        langchain_messages.extend(conversation)
        return langchain_messages
    
    def _convert_to_nat_messages(messages, redact_images=True):
        """Convert OpenAI format messages to dictionaries for OpenAIChatRequest."""
        nat_messages = []
        for msg in messages:
            msg_dict = msg.model_dump() if hasattr(msg, 'model_dump') else dict(msg)
            role = msg_dict.get('role')
            content = msg_dict.get('content')
            
            # Optionally redact images
            if redact_images:
                content = _redact_images_from_content(content)
            
            # Return plain dictionaries, not Message objects
            nat_messages.append({"role": role, "content": content})
        
        return nat_messages
    
    from collections.abc import AsyncGenerator

    from nat.data_models.api_server import ChatResponseChunk

    async def _stream_fn(chat_request: ChatRequest) -> AsyncGenerator[ChatResponseChunk]:
        """Streaming variant: chitchat and vision routes stream tokens as they
        generate (the voice pipeline starts speaking immediately instead of
        waiting for the full reply); the ReAct agent stays non-streaming since
        its trace must be parsed whole.
        """

        try:
            async for chunk in _stream_routes(chat_request):
                yield chunk
        except Exception as e:
            # A silent robot is the worst failure mode: surface errors as speech.
            logger.error(f"RouterAgent(stream): error, returning spoken apology: {e}", exc_info=True)
            yield ChatResponseChunk.create_streaming_chunk(
                "Sorry, something went wrong with that one. Could you try asking again?",
                role="assistant", model="error")
            yield ChatResponseChunk.create_streaming_chunk(None, model="error", finish_reason="stop")

    async def _stream_routes(chat_request) -> AsyncGenerator[ChatResponseChunk]:
        router_response = await router_function.ainvoke(chat_request)
        route = router_response.choices[0].message.content
        logger.info(f"RouterAgent(stream): intent '{route}'")

        # Intermediate chunks must carry finish_reason=None (from_string marks
        # every chunk "stop", which ends the client stream after one chunk).
        if route == "chit_chat":
            langchain_messages = _convert_to_langchain_messages(chat_request.messages, redact_images=True)
            async for chunk in chitchat_llm.astream(langchain_messages):
                content = getattr(chunk, "content", None)
                if content:
                    yield ChatResponseChunk.create_streaming_chunk(content, role="assistant", model="chitchat")
            yield ChatResponseChunk.create_streaming_chunk(None, model="chitchat", finish_reason="stop")
        elif route == "image_understanding":
            langchain_messages = _convert_to_langchain_messages(chat_request.messages, redact_images=False)
            async for chunk in image_llm.astream(langchain_messages):
                content = getattr(chunk, "content", None)
                if content:
                    yield ChatResponseChunk.create_streaming_chunk(content, role="assistant", model="image_understanding")
            yield ChatResponseChunk.create_streaming_chunk(None, model="image_understanding", finish_reason="stop")
        else:
            nat_messages = _convert_to_nat_messages(chat_request.messages, redact_images=True)
            agent_input = {
                "messages": nat_messages,
                "model": chat_request.model if hasattr(chat_request, 'model') else "nemotron",
            }
            agent_response = await agent_function.ainvoke(agent_input)
            content = _clean_agent_reply(agent_response.choices[0].message.content)
            yield ChatResponseChunk.create_streaming_chunk(content, role="assistant", model="agent")
            yield ChatResponseChunk.create_streaming_chunk(None, model="agent", finish_reason="stop")

    # Stream-only: every live client streams (the bot's pipeline always
    # sets stream=true); the non-streaming single_fn path was unreachable
    # dead weight and has been removed.
    yield FunctionInfo.create(
        stream_fn=_stream_fn,
        description="Route chat requests between chitchat and agent based on intent"
    )

