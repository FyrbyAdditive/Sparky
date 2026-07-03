import ast
import logging

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.component_ref import LLMRef

from typing import List, Dict, Any
import json


logger = logging.getLogger(__name__)

# Prompt for the router
TASK_INSTRUCTION = """
You are a helpful assistant designed to find the best suited route.
You are provided with route description within <routes></routes> XML tags:
<routes>

{routes}

</routes>

<conversation>

{conversation}

</conversation>
"""

FORMAT_PROMPT = """
Your task is to decide which route is best suit with user intent on the conversation in <conversation></conversation> XML tags.  Follow the instruction:
1. If the latest intent from user is irrelevant or user intent is full filled, response with other route {"route": "other"}.
2. You must analyze the route descriptions and find the best match route for user latest intent. 
3. You only response the name of the route that best matches the user's request, use the exact name in the <routes></routes>.

Based on your analysis, provide your response in the following JSON formats if you decide to match any route:
{"route": "route_name"} 
"""

# Custom JSON encoder for Pydantic models and non-serializable objects
class PydanticEncoder(json.JSONEncoder):
    def default(self, obj):
        # Handle Pydantic models
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
        # Handle dict-like objects
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        # Handle iterables (except strings)
        if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
            try:
                return list(obj)
            except TypeError:
                pass
        return super().default(obj)

# Helper function to redact images while preserving text context
def redact_images_from_conversation(conversation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove image data for router."""
    redacted = []
    for i, msg in enumerate(conversation):
        msg_copy = msg.copy()
        content = msg_copy.get("content")
    
        # If content is a list (multimodal), process it
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        item_text = item.get("text", "")
                        text = f"<new msg>{item_text} </msg>"
                        text_parts.append(text)
                    elif item.get("type") == "image_url":
                        continue
                        
            # Combine text parts and add image indicator if present
            combined_text = " ".join(text_parts)
            
            msg_copy["content"] = combined_text
        
        redacted.append(msg_copy)
    
    return redacted

def materialize_iterator(obj):
    """Recursively convert ValidatorIterator and other iterables to lists."""
    if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, dict)):
        try:
            return [materialize_iterator(item) for item in obj]
        except TypeError:
            pass
    elif isinstance(obj, dict):
        return {k: materialize_iterator(v) for k, v in obj.items()}
    return obj

# Helper function to create the system prompt for our model
def format_prompt(conversation: List[Dict[str, Any]], routes_json: str):
    """Create the system prompt from the pre-serialized routes JSON."""
    return (
        TASK_INSTRUCTION.format(
            routes=routes_json,
            conversation=json.dumps(conversation, cls=PydanticEncoder)
        )
        + FORMAT_PROMPT
    )

def _parse_route_response(response: str) -> str:
    """Extract the route name; never raise — a bad route must degrade to the
    agent path, not drop the turn (a silent robot is the worst failure)."""
    try:
        return json.loads(response)["route"]
    except (json.JSONDecodeError, TypeError, KeyError):
        pass
    try:
        # Handle single quote format
        return ast.literal_eval(response)["route"]
    except (ValueError, SyntaxError, TypeError, KeyError):
        logger.warning(f"Router: unparseable route response {response!r}; defaulting to 'other'")
        return "other"



class RouterConfig(FunctionBaseConfig, name="router"):
    """Determine the best model given the user intent."""
    llm_name: LLMRef = LLMRef("routing_llm")
    route_config: List[Dict[str, str]] = Field(
        default=[
            {
                "name": "other",
                "description": "Any question that requires careful thought, outside information, image understanding, or tool calling to take actions.",
            },
            {
                "name": "chit_chat",
                "description": "Any simple chit chat, small talk, or casual conversation.",
            },
        ],
        description="List of available intents with their descriptions",
    )

@register_function(config_type=RouterConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def router_fn(config: RouterConfig, builder: Builder):
    """Determine best llm for the route."""

    from nat.data_models.api_server import ChatResponse, ChatResponseChoice, Usage, ChoiceMessage
    from ces_tutorial.openai_chat_request import OpenAIChatRequest as ChatRequest

    import time
    import json


    router_llm = await builder.get_llm(llm_name=config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    route_config = config.route_config
    # the route table is constant for the workflow's lifetime — serialize once
    routes_json = json.dumps(route_config, cls=PydanticEncoder)

    async def get_route_from_conversation(conversation: List[Dict[str, Any]]) -> str:
        """Determine the best route for the conversation (using route llm)."""
        redacted_conversation = redact_images_from_conversation(conversation)
        route_prompt = format_prompt(redacted_conversation, routes_json)

        messages = [
            {"role": "user", "content": route_prompt},
        ]

        try:
            # ainvoke, not invoke: a sync call here blocks the whole event
            # loop for the router round-trip, on every turn
            response = await router_llm.ainvoke(messages)
            response_text = response.content
        except Exception as e:
            logger.error(f"Failed to call remote model: {e}")
            raise

        route = _parse_route_response(response_text)

        return route

    async def _response_fn(chat_request: ChatRequest) -> ChatResponse:  # pyright: ignore[reportUnusedParameter]
        """Determine where to route the request"""

        messages = chat_request.messages

        logger.debug(f"Router: Received {len(messages)} messages")

        if messages:

            last_msg = messages[-1]
            last_msg_dict = last_msg.model_dump() if hasattr(last_msg, 'model_dump') else dict(last_msg)

            last_msg_dict = materialize_iterator(last_msg_dict)

            if logger.isEnabledFor(logging.DEBUG):
                content = last_msg_dict.get('content')
                if isinstance(content, list):
                    logger.debug(f"Router: Last message has list content with {len(content)} items")
                    for i, item in enumerate(content):
                        if isinstance(item, dict):
                            logger.debug(f"Router:   Item {i} - type: {item.get('type')}")
                else:
                    logger.debug(f"Router: Last message content is string, length: {len(str(content))}")

            # Assign a list containing only the last message's dictionary
            messages_dict = [last_msg_dict]

        else:
            # Handle the case where the list of messages is empty
            messages_dict = []
            logger.warning("No messages received in chat request")

        user_intent = await get_route_from_conversation(messages_dict)


        logger.debug(f"User intent: {user_intent}")


        return ChatResponse(
            id="chatcmpl-" + str(int(time.time())),
            object="chat.completion",
            created=int(time.time()),
            model="router",
            choices=[
                ChatResponseChoice(
                    index=0,
                    message=ChoiceMessage(
                        role="assistant",
                        content=user_intent
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

    
    yield FunctionInfo.from_fn(
        _response_fn,
        description="Route between different LLMs based on user message")
