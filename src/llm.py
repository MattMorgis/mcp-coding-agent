import json
from typing import Iterable, List, Type

from anthropic import Anthropic
from anthropic.types import (
    ContentBlock,
    DocumentBlockParam,
    ImageBlockParam,
    Message,
    MessageParam,
    TextBlock,
    TextBlockParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)
from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    EmbeddedResource,
    ImageContent,
    ModelPreferences,
    StopReason,
    TextContent,
    TextResourceContents,
)
from mcp_agent.logging.logger import get_logger

# from mcp_agent import console
# from mcp_agent.agents.agent import HUMAN_INPUT_TOOL_NAME
from mcp_agent.workflows.llm.augmented_llm import (
    AugmentedLLM,
    MCPMessageParam,
    MCPMessageResult,
    ModelT,
    ProviderToMCPConverter,
    RequestParams,
)
from pydantic import BaseModel


class AnthropicAugmentedLLM(AugmentedLLM[MessageParam, Message]):
    """
    The basic building block of agentic systems is an LLM enhanced with augmentations
    such as retrieval, tools, and memory provided from a collection of MCP servers.
    Our current models can actively use these capabilities—generating their own search queries,
    selecting appropriate tools, and determining what information to retain.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            type_converter=AnthropicMCPTypeConverter,
            **kwargs,
        )

        self.provider = "Anthropic"
        # Initialize logger with name if available
        self.logger = get_logger(f"{__name__}.{self.name}" if self.name else __name__)

        self.model_preferences = self.model_preferences or ModelPreferences(
            costPriority=0.3,
            speedPriority=0.4,
            intelligencePriority=0.3,
        )

        default_model = "claude-3-7-sonnet-latest"  # Fallback default

        if self.context.config.anthropic:
            if hasattr(self.context.config.anthropic, "default_model"):
                default_model = self.context.config.anthropic.default_model
        self.default_request_params = self.default_request_params or RequestParams(
            model=default_model,
            modelPreferences=self.model_preferences,
            maxTokens=2048,
            systemPrompt=self.instruction,
            parallel_tool_calls=False,
            max_iterations=10,
            use_history=True,
        )

    async def generate(
        self,
        message,
        request_params: RequestParams | None = None,
        on_iteration_complete=None,
        on_tool_use=None,
        on_tool_result=None,
    ):
        """
        Process a query using an LLM and available tools.
        The default implementation uses Claude as the LLM.
        Override this method to use a different LLM.
        
        Args:
            message: The message to send to the LLM
            request_params: Optional parameters for the request
            on_iteration_complete: Callback function called after each LLM response
                with (iteration_index, llm_response, tool_results_so_far)
            on_tool_use: Callback function called when a tool is about to be used
                with (tool_name, tool_args, tool_use_id)
            on_tool_result: Callback function called after a tool returns results
                with (tool_name, tool_args, tool_result, tool_use_id)
        """
        config = self.context.config
        anthropic = Anthropic(api_key=config.anthropic.api_key)
        messages: List[MessageParam] = []
        params = self.get_request_params(request_params)

        if params.use_history:
            messages.extend(self.history.get())

        if isinstance(message, str):
            messages.append({"role": "user", "content": message})
        elif isinstance(message, list):
            messages.extend(message)
        else:
            messages.append(message)

        response = await self.aggregator.list_tools()
        available_tools: List[ToolParam] = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
            for tool in response.tools
        ]

        responses: List[Message] = []
        tool_results = []
        model = await self.select_model(params)

        for i in range(params.max_iterations):
            if i == params.max_iterations - 1 and responses and responses[-1].stop_reason == "tool_use":
                final_prompt_message = MessageParam(
                    role="user",
                    content="We've reached the maximum number of iterations. "
                    "Please stop using tools now and provide your final comprehensive answer based on all tool results so far."
                    "At the beginning of your response, clearly indicate that your answer may be incomplete due to reaching "
                    "the maximum number of tool usage iterations.",
                )
                messages.append(final_prompt_message)

            arguments = {
                "model": model,
                "max_tokens": params.maxTokens,
                "messages": messages,
                "system": self.instruction or params.systemPrompt,
                "stop_sequences": params.stopSequences,
                "tools": available_tools,
            }

            if params.metadata:
                arguments = {**arguments, **params.metadata}

            self.logger.debug(f"{arguments}")
            self._log_chat_progress(chat_turn=(len(messages) + 1) // 2, model=model)

            executor_result = await self.executor.execute(
                anthropic.messages.create, **arguments
            )

            response = executor_result[0]

            if isinstance(response, BaseException):
                self.logger.error(f"Error: {executor_result}")
                break

            self.logger.debug(
                f"{model} response:",
                data=response,
            )

            response_as_message = self.convert_message_to_message_param(response)
            messages.append(response_as_message)
            responses.append(response)

            # Call the iteration complete callback with the current state
            if on_iteration_complete:
                await on_iteration_complete(i, response, tool_results.copy())

            # Only reset tool_results after the callback has been called
            tool_results = []  

            if response.stop_reason == "end_turn":
                self.logger.debug(
                    f"Iteration {i}: Stopping because finish_reason is 'end_turn'"
                )
                break
            elif response.stop_reason == "stop_sequence":
                # We have reached a stop sequence
                self.logger.debug(
                    f"Iteration {i}: Stopping because finish_reason is 'stop_sequence'"
                )
                break
            elif response.stop_reason == "max_tokens":
                # We have reached the max tokens limit
                self.logger.debug(
                    f"Iteration {i}: Stopping because finish_reason is 'max_tokens'"
                )
                # TODO: saqadri - would be useful to return the reason for stopping to the caller
                break
            else:  # response.stop_reason == "tool_use":
                for content in response.content:
                    if content.type == "tool_use":
                        tool_name = content.name
                        tool_args = content.input
                        tool_use_id = content.id

                        # Notify about the tool use
                        if on_tool_use:
                            await on_tool_use(tool_name, tool_args, tool_use_id)

                        tool_call_request = CallToolRequest(
                            method="tools/call",
                            params=CallToolRequestParams(
                                name=tool_name, arguments=tool_args
                            ),
                        )

                        result = await self.call_tool(
                            request=tool_call_request, tool_call_id=tool_use_id
                        )
                        
                        # Add to the tool results list for the next iteration's callback
                        tool_results.append((tool_name, tool_args, result, tool_use_id))
                        
                        # Notify about the tool result
                        if on_tool_result:
                            await on_tool_result(tool_name, tool_args, result, tool_use_id)

                        # Convert the result to a message and add to the conversation
                        tool_result_message = MessageParam(
                            role="user",
                            content=[
                                ToolResultBlockParam(
                                    type="tool_result",
                                    tool_use_id=tool_use_id,
                                    content=result.content,
                                    is_error=result.isError,
                                )
                            ],
                        )
                        messages.append(tool_result_message)

        if params.use_history:
            self.history.set(messages)

        self._log_chat_finished(model=model)

        return responses

    async def generate_str(
        self,
        message,
        request_params: RequestParams | None = None,
        on_new_text=None,  # Callback for getting text updates
    ) -> str:
        """
        Process a query using an LLM and available tools and return the final string result.
        Optionally provides streaming updates via the on_new_text callback.
        
        Args:
            message: The message to send to the LLM
            request_params: Optional parameters for the request
            on_new_text: Callback function that will be called with string updates
                as they become available
                
        Returns:
            The complete string response
        """
        final_text: List[str] = []
        
        # Define callbacks for streaming
        async def handle_iteration(iteration, response, tool_results):
            chunk_text = []
            for content in response.content:
                if content.type == "text":
                    chunk_text.append(content.text)
            
            if chunk_text and on_new_text:
                await on_new_text("\n".join(chunk_text), "llm_response")
        
        async def handle_tool_use(tool_name, tool_args, tool_use_id):
            if on_new_text:
                tool_text = f"[Calling tool {tool_name} with args {json.dumps(tool_args, indent=2)}]"
                await on_new_text(tool_text, "tool_use")
        
        async def handle_tool_result(tool_name, tool_args, result, tool_use_id):
            if on_new_text:
                # Extract text from tool result
                result_text = []
                for content in result.content:
                    if hasattr(content, "text"):
                        result_text.append(content.text)
                
                tool_result_str = f"[Tool {tool_name} returned: {' '.join(result_text)}]"
                await on_new_text(tool_result_str, "tool_result")
        
        # Call generate with our streaming callbacks
        responses: List[Message] = await self.generate(
            message=message,
            request_params=request_params,
            on_iteration_complete=handle_iteration,
            on_tool_use=handle_tool_use,
            on_tool_result=handle_tool_result,
        )

        # Collect the complete response
        for response in responses:
            for content in response.content:
                if content.type == "text":
                    final_text.append(content.text)
                elif content.type == "tool_use":
                    final_text.append(
                        f"[Calling tool {content.name} with args {content.input}]"
                    )

        return "\n".join(final_text)

    async def generate_structured(
        self,
        message,
        response_model: Type[ModelT],
        request_params: RequestParams | None = None,
        on_new_text=None,  # Pass through the streaming capability
    ) -> ModelT:
        # First we invoke the LLM to generate a string response
        # We need to do this in a two-step process because Instructor doesn't
        # know how to invoke MCP tools via call_tool, so we'll handle all the
        # processing first and then pass the final response through Instructor
        import instructor

        response = await self.generate_str(
            message=message,
            request_params=request_params,
            on_new_text=on_new_text,  # Pass through the streaming capability
        )

        # Next we pass the text through instructor to extract structured data
        client = instructor.from_anthropic(
            Anthropic(api_key=self.context.config.anthropic.api_key),
        )

        params = self.get_request_params(request_params)
        model = await self.select_model(params)

        # Extract structured data from natural language
        structured_response = client.chat.completions.create(
            model=model,
            response_model=response_model,
            messages=[{"role": "user", "content": response}],
            max_tokens=params.maxTokens,
        )

        return structured_response

    @classmethod
    def convert_message_to_message_param(
        cls, message: Message, **kwargs
    ) -> MessageParam:
        """Convert a response object to an input parameter object to allow LLM calls to be chained."""
        content = []

        for content_block in message.content:
            if content_block.type == "text":
                content.append(TextBlockParam(type="text", text=content_block.text))
            elif content_block.type == "tool_use":
                content.append(
                    ToolUseBlockParam(
                        type="tool_use",
                        name=content_block.name,
                        input=content_block.input,
                        id=content_block.id,
                    )
                )

        return MessageParam(role="assistant", content=content, **kwargs)

    def message_param_str(self, message: MessageParam) -> str:
        """Convert an input message to a string representation."""

        if message.get("content"):
            content = message["content"]
            if isinstance(content, str):
                return content
            else:
                final_text: List[str] = []
                for block in content:
                    if block.text:
                        final_text.append(str(block.text))
                    else:
                        final_text.append(str(block))

                return "\n".join(final_text)

        return str(message)

    def message_str(self, message: Message) -> str:
        """Convert an output message to a string representation."""
        content = message.content

        if content:
            if isinstance(content, list):
                final_text: List[str] = []
                for block in content:
                    if block.text:
                        final_text.append(str(block.text))
                    else:
                        final_text.append(str(block))

                return "\n".join(final_text)
            else:
                return str(content)

        return str(message)


class AnthropicMCPTypeConverter(ProviderToMCPConverter[MessageParam, Message]):
    """
    Convert between Anthropic and MCP types.
    """

    @classmethod
    def from_mcp_message_result(cls, result: MCPMessageResult) -> Message:
        # MCPMessageResult -> Message
        if result.role != "assistant":
            raise ValueError(
                f"Expected role to be 'assistant' but got '{result.role}' instead."
            )

        return Message(
            role="assistant",
            type="message",
            content=[mcp_content_to_anthropic_content(result.content)],
            model=result.model,
            stop_reason=mcp_stop_reason_to_anthropic_stop_reason(result.stopReason),
            id=result.id or None,
            usage=result.usage or None,
            # TODO: should we push extras?
        )

    @classmethod
    def to_mcp_message_result(cls, result: Message) -> MCPMessageResult:
        # Message -> MCPMessageResult

        contents = anthropic_content_to_mcp_content(result.content)
        if len(contents) > 1:
            raise NotImplementedError(
                "Multiple content elements in a single message are not supported in MCP yet"
            )
        mcp_content = contents[0]

        return MCPMessageResult(
            role=result.role,
            content=mcp_content,
            model=result.model,
            stopReason=anthropic_stop_reason_to_mcp_stop_reason(result.stop_reason),
            # extras for Message fields
            **result.model_dump(exclude={"role", "content", "model", "stop_reason"}),
        )

    @classmethod
    def from_mcp_message_param(cls, param: MCPMessageParam) -> MessageParam:
        # MCPMessageParam -> MessageParam
        extras = param.model_dump(exclude={"role", "content"})
        return MessageParam(
            role=param.role,
            content=[mcp_content_to_anthropic_content(param.content)],
            **extras,
        )

    @classmethod
    def to_mcp_message_param(cls, param: MessageParam) -> MCPMessageParam:
        # Implement the conversion from ChatCompletionMessage to MCP message param

        contents = anthropic_content_to_mcp_content(param.content)

        # TODO: saqadri - the mcp_content can have multiple elements
        # while sampling message content has a single content element
        # Right now we error out if there are > 1 elements in mcp_content
        # We need to handle this case properly going forward
        if len(contents) > 1:
            raise NotImplementedError(
                "Multiple content elements in a single message are not supported"
            )
        mcp_content = contents[0]

        return MCPMessageParam(
            role=param.role,
            content=mcp_content,
            **typed_dict_extras(param, ["role", "content"]),
        )


def mcp_content_to_anthropic_content(
    content: TextContent | ImageContent | EmbeddedResource,
) -> ContentBlock:
    if isinstance(content, TextContent):
        return TextBlock(type=content.type, text=content.text)
    elif isinstance(content, ImageContent):
        # Best effort to convert an image to text (since there's no ImageBlock)
        return TextBlock(type="text", text=f"{content.mimeType}:{content.data}")
    elif isinstance(content, EmbeddedResource):
        if isinstance(content.resource, TextResourceContents):
            return TextBlock(type="text", text=content.resource.text)
        else:  # BlobResourceContents
            return TextBlock(
                type="text", text=f"{content.resource.mimeType}:{content.resource.blob}"
            )
    else:
        # Last effort to convert the content to a string
        return TextBlock(type="text", text=str(content))


def anthropic_content_to_mcp_content(
    content: str
    | Iterable[
        TextBlockParam
        | ImageBlockParam
        | ToolUseBlockParam
        | ToolResultBlockParam
        | DocumentBlockParam
        | ContentBlock
    ],
) -> List[TextContent | ImageContent | EmbeddedResource]:
    mcp_content = []

    if isinstance(content, str):
        mcp_content.append(TextContent(type="text", text=content))
    else:
        for block in content:
            if block.type == "text":
                mcp_content.append(TextContent(type="text", text=block.text))
            elif block.type == "image":
                raise NotImplementedError("Image content conversion not implemented")
            elif block.type == "tool_use":
                # Best effort to convert a tool use to text (since there's no ToolUseContent)
                mcp_content.append(
                    TextContent(
                        type="text",
                        text=to_string(block),
                    )
                )
            elif block.type == "tool_result":
                # Best effort to convert a tool result to text (since there's no ToolResultContent)
                mcp_content.append(
                    TextContent(
                        type="text",
                        text=to_string(block),
                    )
                )
            elif block.type == "document":
                raise NotImplementedError("Document content conversion not implemented")
            else:
                # Last effort to convert the content to a string
                mcp_content.append(TextContent(type="text", text=str(block)))

    return mcp_content


def mcp_stop_reason_to_anthropic_stop_reason(stop_reason: StopReason):
    if not stop_reason:
        return None
    elif stop_reason == "endTurn":
        return "end_turn"
    elif stop_reason == "maxTokens":
        return "max_tokens"
    elif stop_reason == "stopSequence":
        return "stop_sequence"
    elif stop_reason == "toolUse":
        return "tool_use"
    else:
        return stop_reason


def anthropic_stop_reason_to_mcp_stop_reason(stop_reason: str) -> StopReason:
    if not stop_reason:
        return None
    elif stop_reason == "end_turn":
        return "endTurn"
    elif stop_reason == "max_tokens":
        return "maxTokens"
    elif stop_reason == "stop_sequence":
        return "stopSequence"
    elif stop_reason == "tool_use":
        return "toolUse"
    else:
        return stop_reason


def to_string(obj: BaseModel | dict) -> str:
    if isinstance(obj, BaseModel):
        return obj.model_dump_json()
    else:
        return json.dumps(obj)


def typed_dict_extras(d: dict, exclude: List[str]):
    extras = {k: v for k, v in d.items() if k not in exclude}
    return extras
