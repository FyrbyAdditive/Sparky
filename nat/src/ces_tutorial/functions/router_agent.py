import logging

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.component_ref import LLMRef, FunctionRef

logger = logging.getLogger(__name__)


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
    
    from nat.data_models.api_server import ChatResponse, ChatResponseChoice, Usage, ChoiceMessage
    from ces_tutorial.openai_chat_request import OpenAIChatRequest as ChatRequest
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    import time
    
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
        """Convert OpenAI format messages to LangChain messages."""
        langchain_messages = []
        for msg in messages:
            msg_dict = msg.model_dump() if hasattr(msg, 'model_dump') else dict(msg)
            role = msg_dict.get('role')
            content = msg_dict.get('content')
            
            # Optionally redact images
            if redact_images:
                content = _redact_images_from_content(content)
            
            # Create appropriate LangChain message based on role
            if role == 'system':
                langchain_messages.append(SystemMessage(content=content))
            elif role == 'user':
                langchain_messages.append(HumanMessage(content=content))
            elif role == 'assistant':
                langchain_messages.append(AIMessage(content=content))
        
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
    
    def _create_chat_response(content, model_name):
        """Create a ChatResponse from LLM content."""
        return ChatResponse(
            id="chatcmpl-" + str(int(time.time())),
            object="chat.completion",
            created=int(time.time()),
            model=model_name,
            choices=[
                ChatResponseChoice(
                    index=0,
                    message=ChoiceMessage(
                        role="assistant",
                        content=content
                    ),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0
            )
        )
    
    def _log_message_details(messages, prefix="RouterAgent"):
        """Log detailed information about messages."""
        for idx, msg in enumerate(messages):
            msg_dict = msg.model_dump() if hasattr(msg, 'model_dump') else dict(msg)
            content = msg_dict.get('content')
            content_type = type(content).__name__
            
            if isinstance(content, list):
                logger.warn(f"{prefix}: Message {idx} - role: {msg_dict.get('role')}, content is list with {len(content)} items")
                for i, item in enumerate(content):
                    if isinstance(item, dict):
                        logger.warn(f"{prefix}:   Item {i} - type: {item.get('type')}, keys: {list(item.keys())}")
                    else:
                        logger.warn(f"{prefix}:   Item {i} - {type(item).__name__}")
            else:
                logger.warn(f"{prefix}: Message {idx} - role: {msg_dict.get('role')}, content type: {content_type}, length: {len(str(content)) if content else 0}")
    
    async def _response_fn(chat_request: ChatRequest) -> ChatResponse:
        """Route the request based on intent."""
        
        try:
            logger.warn(f"RouterAgent: Processing request with {len(chat_request.messages)} messages")
            
            # Log message details to check for images
            _log_message_details(chat_request.messages)
            
            # Step 1: Call the router to determine intent
            logger.warn("RouterAgent: Calling router to determine intent...")
            try:
                router_response = await router_function.ainvoke(chat_request)
                logger.warn(f"RouterAgent: Router response received: {type(router_response)}")
            except Exception as e:
                logger.error(f"RouterAgent: Error calling router function: {e}", exc_info=True)
                raise
            
            # Extract the route from the router response
            try:
                route = router_response.choices[0].message.content
                logger.warn(f"RouterAgent: Router determined intent as '{route}'")
            except Exception as e:
                logger.error(f"RouterAgent: Error extracting route from response: {e}", exc_info=True)
                logger.error(f"RouterAgent: Router response structure: {router_response}")
                raise
            
            # Step 2: Route based on the intent
            if route == "chit_chat":
                logger.warn("RouterAgent: Routing to chitchat LLM")
                
                try:
                    # Convert messages to LangChain format and redact images
                    langchain_messages = _convert_to_langchain_messages(chat_request.messages, redact_images=True)
                    logger.warn(f"RouterAgent: Converted {len(langchain_messages)} messages for chitchat LLM (images redacted)")
                    
                    # Call the chitchat LLM
                    response = chitchat_llm.invoke(langchain_messages)
                    logger.warn(f"RouterAgent: Chitchat LLM response received: {type(response)}")
                    
                    # Extract content and create response
                    content = response.content if hasattr(response, 'content') else str(response)
                    return _create_chat_response(content, "chitchat")
                    
                except Exception as e:
                    logger.error(f"RouterAgent: Error in chitchat path: {e}", exc_info=True)
                    raise
            
            elif route == "image_understanding":
                logger.warn("RouterAgent: Routing to image understanding LLM")
                
                try:
                    # Convert messages to LangChain format, preserving images
                    langchain_messages = _convert_to_langchain_messages(chat_request.messages, redact_images=False)
                    logger.warn(f"RouterAgent: Converted {len(langchain_messages)} messages for image LLM")
                    
                    # Log to verify images are present
                    for idx, msg in enumerate(langchain_messages):
                        content = msg.content
                        if isinstance(content, list):
                            logger.warn(f"RouterAgent: [IMAGE PATH] Message {idx} has list content with {len(content)} items")
                            for i, item in enumerate(content):
                                if isinstance(item, dict) and item.get('type') == 'image_url':
                                    logger.warn(f"RouterAgent: [IMAGE PATH] Found image_url at message {idx}, item {i}")
                    
                    # Call the image LLM
                    response = image_llm.invoke(langchain_messages)
                    logger.warn(f"RouterAgent: Image LLM response received: {type(response)}")
                    
                    # Extract content and create response
                    content = response.content if hasattr(response, 'content') else str(response)
                    return _create_chat_response(content, "image_understanding")
                    
                except Exception as e:
                    logger.error(f"RouterAgent: Error in image understanding path: {e}", exc_info=True)
                    raise
                    
            else:  # route == "other" or any other value
                logger.warn(f"RouterAgent: Routing to agent function for '{route}' intent")
                
                try:
                    # Convert messages to dict format
                    # NOTE: Set redact_images=False if you want the agent to see images
                    # (assuming you have a multimodal LLM backing the agent).
                    nat_messages = _convert_to_nat_messages(chat_request.messages, redact_images=True)
                    logger.warn(f"RouterAgent: Converted {len(nat_messages)} messages for agent")
                    
                    # Manually construct the input dictionary for the agent.
                    # We pass a dict that matches the standard ChatRequestOrMessage structure.
                    # The converter in register.py will handle turning this into OpenAIChatRequest.
                    agent_input = {
                        "messages": nat_messages,
                        "model": chat_request.model if hasattr(chat_request, 'model') else "nemotron"
                    }
                    
                    # Call the agent function with the dict
                    agent_response = await agent_function.ainvoke(agent_input)
                    logger.warn(f"RouterAgent: Agent response received: {type(agent_response)}")
                    return agent_response
                    
                except Exception as e:
                    logger.error(f"RouterAgent: Error in agent path: {e}", exc_info=True)
                    raise
                    
        except Exception as e:
            logger.error(f"RouterAgent: Top-level error in _response_fn: {e}", exc_info=True)
            raise

    from collections.abc import AsyncGenerator

    from nat.data_models.api_server import ChatResponseChunk

    async def _stream_fn(chat_request: ChatRequest) -> AsyncGenerator[ChatResponseChunk]:
        """Streaming variant: chitchat and vision routes stream tokens as they
        generate (the voice pipeline starts speaking immediately instead of
        waiting for the full reply); the ReAct agent stays non-streaming since
        its trace must be parsed whole.
        """

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
            content = agent_response.choices[0].message.content
            yield ChatResponseChunk.create_streaming_chunk(content, role="assistant", model="agent")
            yield ChatResponseChunk.create_streaming_chunk(None, model="agent", finish_reason="stop")

    yield FunctionInfo.create(
        single_fn=_response_fn,
        stream_fn=_stream_fn,
        description="Route chat requests between chitchat and agent based on intent"
    )

