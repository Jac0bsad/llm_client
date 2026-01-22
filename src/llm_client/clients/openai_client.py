import json
import logging as logger
from pathlib import Path
from threading import Lock
from typing import (
    AsyncGenerator,
    Callable,
    Generator,
    Optional,
    Any,
    Dict,
    Type,
    TypeVar,
)

import yaml
from pydantic import BaseModel
from openai import OpenAI, AsyncOpenAI
from openai.types import CompletionUsage, Completion
from openai.types.responses import ResponseUsage

config_file_path = Path(__file__).parent.parent / "conf" / "llm.yaml"
T = TypeVar("T", bound=BaseModel)


class ToolCallStart(BaseModel):
    tool_call_id: str
    tool_name: str
    tool_arguments: dict


class ToolCallEnd(BaseModel):
    tool_call_id: str
    tool_name: str
    tool_arguments: dict
    tool_result: str


class StreamToolCallResponse(BaseModel):
    reasoning_content: Optional[str] = None
    content: Optional[str] = None
    tool_call_start: Optional[ToolCallStart] = None
    tool_call_end: Optional[ToolCallEnd] = None


def str_to_json(
    content: str, output_type: Optional[Type[T]] = None
) -> dict | list[dict] | T | list[T]:
    """
    将大模型输出的json格式字符串进行处理后加载为Python对象
    """
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    data = json.loads(content)
    if output_type:
        if isinstance(data, list):
            return [output_type.model_validate(item) for item in data]
        return output_type.model_validate(data)
    return data


class LLMConfig(BaseModel):
    base_url: str
    api_key: str
    model: str
    input_cost: float
    output_cost: float
    input_cost_cache_hit: float


def _read_yaml_config(path: Path = config_file_path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if (
        not isinstance(data, dict)
        or "models" not in data
        or not isinstance(data["models"], dict)
    ):
        raise ValueError(f"Invalid LLM config at {path}: missing 'models' mapping")
    return data


def get_llm_config(
    model_name: Optional[str] = None, path: Path = config_file_path
) -> LLMConfig:
    cfg = _read_yaml_config(path)
    name = model_name or cfg.get("default_model")
    if not name:
        raise ValueError(
            "No model name provided and 'default_model' is not set in the config file"
        )

    models = cfg["models"]
    if name not in models:
        available = ", ".join(sorted(models.keys()))
        raise KeyError(
            f"Model '{name}' not found in config. Available models: {available}"
        )

    details = models[name] or {}

    # Translate YAML keys -> LLMConfig fields. Validate required fields.
    base_url = details.get("api_base") or details.get("base_url")
    if not base_url:
        raise ValueError(
            f"Model '{name}' is missing required key 'api_base' (or 'base_url')"
        )
    api_key = details.get("api_key")
    if not api_key:
        raise ValueError(f"Model '{name}' is missing required key 'api_key'")

    return LLMConfig(
        base_url=base_url,
        api_key=api_key,
        model=details.get("model_name") or details.get("model") or name,
        input_cost=float(details.get("input_cost", 0.0)),
        output_cost=float(details.get("output_cost", 0.0)),
        input_cost_cache_hit=float(details.get("input_cost_cache_hit", 0.0)),
    )


class OpenAIClient:
    def __init__(self):
        """初始化OpenAI客户端，设置token使用统计"""
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
        }
        self.lock = Lock()
        self.cost = 0.0
        self.input_cost = 0.0  # 每百万token计费
        self.output_cost = 0.0  # 每百万token计费
        self.input_cost_cache_hit = 0.0  # 每百万token计费

    def _get_client_config(self, model: Optional[str] = "deepseek"):
        """获取客户端配置"""
        cfg = get_llm_config(model)
        api_base, api_key, model_name = cfg.base_url, cfg.api_key, cfg.model
        self.input_cost = cfg.input_cost
        self.output_cost = cfg.output_cost
        self.input_cost_cache_hit = cfg.input_cost_cache_hit
        return api_base, api_key, model_name

    def _update_token_usage(self, usage: CompletionUsage | ResponseUsage) -> None:
        """更新累计的token使用量, 线程安全"""
        one_million = 1_000_000
        with self.lock:
            if isinstance(usage, CompletionUsage):
                cached_tokens = usage.prompt_tokens_details.cached_tokens
                self.token_usage["cached_tokens"] += cached_tokens
                self.token_usage["prompt_tokens"] += usage.prompt_tokens - cached_tokens
                self.token_usage["completion_tokens"] += usage.completion_tokens
                self.token_usage["total_tokens"] += usage.total_tokens
            else:
                cached_tokens = usage.input_tokens_details.cached_tokens
                self.token_usage["cached_tokens"] += cached_tokens
                self.token_usage["prompt_tokens"] += usage.input_tokens - cached_tokens
                self.token_usage["completion_tokens"] += usage.output_tokens
                self.token_usage["total_tokens"] += usage.total_tokens
            cost = (
                self.token_usage["prompt_tokens"] * self.input_cost
                + self.token_usage["completion_tokens"] * self.output_cost
                + self.token_usage["cached_tokens"] * self.input_cost_cache_hit
            ) / one_million
            self.cost = cost

    def _process_response(self, response: Completion) -> str:
        usage = response.usage
        # 累计token消耗总量
        if usage:
            logger.info(usage)
            self._update_token_usage(usage)
        return response.choices[0].message.content

    def send_messages(
        self,
        messages: list[dict],
        model: Optional[str] = "deepseek",
        extra_body: Optional[dict] = None,
    ) -> str:
        """
        发送消息到大模型，并返回响应
        Args:
            messages: 消息列表
            model: 模型名称，默认为"deepseek"
            extra_body: 额外的请求体，默认为None
        Return:
            纯文本响应内容
        """
        api_base, api_key, model_name = self._get_client_config(model)
        client = OpenAI(api_key=api_key, base_url=api_base)

        response = client.chat.completions.create(
            model=model_name, messages=messages, stream=False, extra_body=extra_body
        )

        return self._process_response(response)

    async def send_messages_async(
        self,
        messages: list[dict],
        model: Optional[str] = "deepseek",
        extra_body: Optional[dict] = None,
    ) -> str:
        """
        协程发送消息到大模型，并返回响应
        Args:
            messages: 消息列表
            model: 模型名称，默认为"deepseek"
            extra_body: 额外的请求体，默认为None
        Return:
            纯文本响应内容
        """
        api_base, api_key, model_name = self._get_client_config(model)
        async with AsyncOpenAI(api_key=api_key, base_url=api_base) as client:
            response = await client.chat.completions.create(
                model=model_name, messages=messages, stream=False, extra_body=extra_body
            )

        return self._process_response(response)

    def get_json_response(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        output_type: Optional[Type[T]] = None,
        extra_body: Optional[dict] = None,
        use_response: bool = False,
    ) -> dict | list[dict] | list[T] | T:
        """
        发送消息到大模型，并返回json格式的响应
        Args:
            messages: 消息列表
            model: 模型名称，默认为"deepseek"
            output_type: 基于Pydantic的模型，用于验证和转换响应内容
            extra_body: 额外的请求体，默认为None
            use_response: 是否使用response api，默认为False，如果使用，必须指定output_type
        Return:
            json.loads()后的响应内容
        """
        api_base, api_key, model_name = self._get_client_config(model)
        client = OpenAI(api_key=api_key, base_url=api_base)
        if use_response:
            if not output_type:
                raise ValueError("output_type is required when use_response is True")
            response = client.responses.parse(
                model=model_name,
                input=messages,
                extra_body=extra_body,
                text_format=output_type,
            )
            usage = response.usage
            logger.info(usage)
            self._update_token_usage(usage)
            return response.output_parsed

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=False,
            response_format={"type": "json_object"},
            extra_body=extra_body,
        )

        content = self._process_response(response)

        return str_to_json(content, output_type)

    async def get_json_response_async(
        self,
        messages: list[dict],
        model: Optional[str] = "deepseek",
        output_type: Optional[Type[T]] = None,
        extra_body: Optional[dict] = None,
        use_response: bool = False,
    ) -> dict | list[dict] | list[T] | T:
        """
        协程发送消息到大模型，并返回json格式的响应
        Args:
            messages: 消息列表
        Return:
            json.loads()后的响应内容
        """
        api_base, api_key, model_name = self._get_client_config(model)
        async with AsyncOpenAI(api_key=api_key, base_url=api_base) as client:
            if use_response:
                if not output_type:
                    raise ValueError(
                        "output_type is required when use_response is True"
                    )
                response = await client.responses.parse(
                    model=model_name,
                    input=messages,
                    extra_body=extra_body,
                    text_format=output_type,
                )
                usage = response.usage
                logger.info(usage)
                self._update_token_usage(usage)
                return response.output_parsed

            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=False,
                response_format={"type": "json_object"},
                extra_body=extra_body,
            )

        content = self._process_response(response)

        return str_to_json(content, output_type)

    def send_messages_stream(
        self,
        messages: list[dict],
        config_name: Optional[str] = "deepseek",
        extra_body: Optional[dict] = None,
        response_format: Optional[dict] = None,
        stop: Optional[list[str]] = None,
    ) -> Generator[str, None, str]:
        """
        发送消息到大模型，并返回流式响应，处理内容过长导致的截断
        Args:
            messages: 消息列表
            config_name: 配置名称，默认为"deepseek"
            extra_body: 额外的请求体，默认为None
            response_format: 响应格式，默认为None
            stop: 停止词列表，默认为None
        Returns:
            Generator[str, None, str]: 流式响应生成器
        """
        messages = messages.copy()  # 防止改变原变量
        api_base, api_key, model_name = self._get_client_config(config_name)
        client = OpenAI(api_key=api_key, base_url=api_base)
        full_response = ""
        finish_reason = "length"
        while finish_reason != "stop" or full_response == "":
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    stream=True,
                    stream_options={"include_usage": True},
                    response_format=response_format,
                    stop=stop,
                    extra_body=extra_body,
                )

                for chunk in response:
                    if hasattr(chunk, "usage") and chunk.usage:  # 检查是否有 usage 信息
                        logger.info(chunk.usage)
                        usage = chunk.usage
                        self._update_token_usage(usage)
                    else:  # 如果没有 usage，则处理 choices 内容
                        if chunk.choices and chunk.choices[0].delta.content is not None:
                            yield chunk.choices[0].delta.content
                            full_response += chunk.choices[0].delta.content
                            # print(chunk.choices[0].delta.content, end='')

                        if chunk.choices and chunk.choices[0].finish_reason:
                            finish_reason = chunk.choices[0].finish_reason
                            logger.info(chunk.choices[0].finish_reason)

                messages.append(
                    {"role": "assistant", "content": full_response, "prefix": True}
                )
            except Exception as e:
                logger.error("Error during streaming: %s", e)
                break

        return full_response

    def send_messages_stream_dict(
        self,
        messages: list[dict],
        config_name: Optional[str] = "deepseek",
        extra_body: Optional[dict] = None,
        response_format: Optional[dict] = None,
    ) -> Generator[dict, None, dict]:
        """
        发送消息到大模型，并以dict类型返回流式响应，处理内容过长导致的截断
        Args:
            messages: 消息列表
            config_name: 配置名称，默认为'deepseek'
            extra_body: 额外的请求体，默认为None
            response_format: 响应格式，默认为None
        Returns:
            Generator[dict, None, dict]: 流式响应生成器
            {"reasoning_content": "reasoning_content"}
            {"content": "content"}
        """
        messages = messages.copy()  # 防止改变原变量

        api_base, api_key, model_name = self._get_client_config(config_name)

        client = OpenAI(api_key=api_key, base_url=api_base)
        full_response = ""
        finish_reason = "length"
        while finish_reason != "stop" or full_response == "":
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
                response_format=response_format,
                extra_body=extra_body,
            )

            for chunk in response:
                if hasattr(chunk, "usage") and chunk.usage:  # 检查是否有 usage 信息
                    logger.info(chunk.usage)
                    usage = chunk.usage
                    self._update_token_usage(usage)
                else:  # 如果没有 usage，则处理 choices 内容
                    if chunk.choices and hasattr(chunk.choices[0].delta, "content"):
                        yield {"content": chunk.choices[0].delta.content}
                        full_response += chunk.choices[0].delta.content

                    if chunk.choices and hasattr(
                        chunk.choices[0].delta, "reasoning_content"
                    ):
                        yield {
                            "reasoning_content": chunk.choices[
                                0
                            ].delta.reasoning_content
                        }
                        full_response += chunk.choices[0].delta.reasoning_content

                    if chunk.choices and chunk.choices[0].finish_reason:
                        finish_reason = chunk.choices[0].finish_reason
                        logger.info(chunk.choices[0].finish_reason)

            messages.append(
                {"role": "assistant", "content": full_response, "prefix": True}
            )

    async def send_messages_stream_with_tool_call(
        self,
        messages: list[dict],
        tools: list[dict],
        call_tool_func: Callable,
        config_name: str = "deepseek",
        extra_body: dict = None,
        stop: list[str] = None,
        parallel_tool_calls: bool = True,
    ) -> AsyncGenerator[StreamToolCallResponse, None]:
        """
        流式向大模型发送消息，并处理工具调用
        直到没有新的工具调用请求
        Args:
            messages: 消息列表
            tools: 工具列表
            call_tool_func: 工具调用函数
            config_name: 配置名称，默认为'deepseek'
            reasoning: 是否启用推理，默认为False
            stop: 停止条件，默认为None
        """
        api_base, api_key, model_name = self._get_client_config(config_name)
        messages = messages.copy()  # 防止改变原变量
        content_all = ""

        async with AsyncOpenAI(api_key=api_key, base_url=api_base) as client:
            while True:
                logger.info(
                    "Sending messages to LLM. Current message count: %d",
                    len(messages),
                )
                # 考虑控制messages的大小
                # logger.info(json.dumps(messages, ensure_ascii=False, indent=2))

                response = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    tools=tools,
                    stream=True,
                    stream_options={"include_usage": True},
                    tool_choice="auto",  # None, auto, required
                    stop=stop,
                    parallel_tool_calls=parallel_tool_calls,
                    extra_body=extra_body,
                )

                current_round_content = ""
                # 使用字典来重构工具调用，以其索引为键
                tool_call_deltas_by_index = {}

                async for chunk in response:
                    # logger.info(chunk)

                    if hasattr(chunk, "usage") and chunk.usage:  # 检查是否有 usage 信息
                        usage = chunk.usage
                        self._update_token_usage(usage)
                    else:  # 如果没有 usage，则处理 choices 内容
                        delta = chunk.choices[0].delta
                        if hasattr(delta, "reasoning_content"):
                            yield StreamToolCallResponse(
                                reasoning_content=delta.reasoning_content
                            )

                        if delta.content:
                            current_round_content += delta.content
                            yield StreamToolCallResponse(content=delta.content)
                            # logger.info(delta.content)

                        if delta.tool_calls:
                            for tool_call_delta in delta.tool_calls:
                                index = tool_call_delta.index

                                if index not in tool_call_deltas_by_index:
                                    # 初始化工具调用信息
                                    tool_call_deltas_by_index[index] = {
                                        "id": None,
                                        "type": "function",
                                        "function": {"name": None, "arguments": ""},
                                    }

                                # 累加流式工具调用内容
                                if tool_call_delta.id:
                                    tool_call_deltas_by_index[index][
                                        "id"
                                    ] = tool_call_delta.id
                                if tool_call_delta.function:
                                    if tool_call_delta.function.name:
                                        tool_call_deltas_by_index[index]["function"][
                                            "name"
                                        ] = tool_call_delta.function.name
                                    if tool_call_delta.function.arguments:
                                        tool_call_deltas_by_index[index]["function"][
                                            "arguments"
                                        ] += tool_call_delta.function.arguments

                # 整理工具调用记录
                assistant_tool_calls_reconstructed = []
                if tool_call_deltas_by_index:
                    for _idx in sorted(
                        tool_call_deltas_by_index.keys()
                    ):  # 按照工具调用顺序排序
                        assistant_tool_calls_reconstructed.append(
                            tool_call_deltas_by_index[_idx]
                        )

                # 本轮的回答内容
                logger.info("Assistant turn content: %s", current_round_content)
                if assistant_tool_calls_reconstructed:
                    logger.info(
                        "Assistant turn tool calls: %s",
                        json.dumps(
                            assistant_tool_calls_reconstructed,
                            indent=2,
                            ensure_ascii=False,
                        ),
                    )

                # 构建assistant的消息，将回答内和工具调用记录放入消息列表中
                assistant_message = {"role": "assistant"}
                has_content = bool(
                    current_round_content and current_round_content.strip()
                )
                has_tool_calls = bool(assistant_tool_calls_reconstructed)

                if has_content:
                    assistant_message["content"] = current_round_content

                if has_tool_calls:
                    assistant_message["tool_calls"] = assistant_tool_calls_reconstructed

                if has_content or has_tool_calls:
                    messages.append(assistant_message)
                else:
                    # 内容和工具调用都为空
                    logger.warning("LLM response was empty for this turn. ")
                    content_all += current_round_content
                    return

                # 没有进一步的工具调用，说明回答结束
                if not has_tool_calls:
                    logger.info(
                        "LLM processing finished. No more tool calls requested."
                    )
                    content_all += current_round_content
                    return

                # 如果有工具调用，执行工具
                logger.info("LLM requested tool calls. Executing now.")
                for tool_call_obj in assistant_tool_calls_reconstructed:
                    tool_call_id = tool_call_obj["id"]
                    tool_name = tool_call_obj["function"]["name"]
                    arguments_str = tool_call_obj["function"]["arguments"]

                    # 检查工具参数是否完整，arguments_str可能为空
                    if not all([tool_call_id, tool_name]):
                        logger.error(
                            "Malformed tool call object: ID or name missing. Skipping. Object: %s",
                            tool_call_obj,
                        )
                        # 把错误信息放入消息列表
                        if not tool_call_id:
                            idx = assistant_tool_calls_reconstructed.index(
                                tool_call_obj
                            )
                            unknown_id_str = f"unknown_id_for_index_{idx}"
                            tool_id = unknown_id_str
                        else:
                            tool_id = tool_call_id
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_id,
                                "content": (
                                    "Error: Tool call information was incomplete "
                                    "(ID or name missing). Cannot execute."
                                ),
                            }
                        )
                        yield StreamToolCallResponse(
                            tool_call_end=ToolCallEnd(
                                tool_call_id=tool_id,
                                tool_name=tool_name or "unknown",
                                tool_arguments={},
                                tool_result="工具调用失败，参数格式不正确",
                            )
                        )
                        continue

                    logger.info(
                        "Executing tool: %s (ID: %s) with arguments: %s",
                        tool_name,
                        tool_call_id,
                        arguments_str,
                    )

                    parsed_arguments = json.loads(arguments_str)

                    yield StreamToolCallResponse(
                        tool_call_start=ToolCallStart(
                            tool_call_id=tool_call_id,
                            tool_name=tool_name,
                            tool_arguments=parsed_arguments,
                        )
                    )

                    try:
                        tool_result = await call_tool_func(tool_name, parsed_arguments)
                    except json.JSONDecodeError as e:
                        logger.exception(
                            "JSON decoding error for tool %s (ID: %s) arguments '%s': %s",
                            tool_name,
                            tool_call_id,
                            arguments_str,
                            e,
                        )
                        tool_result = (
                            f"Error: Tool {tool_name} arguments were not valid JSON. "
                            f"Error: {e}. Arguments received: {arguments_str}"
                        )
                    except Exception as e:
                        logger.exception(
                            "Error calling tool %s (ID: %s): %s",
                            tool_name,
                            tool_call_id,
                            e,
                        )
                        tool_result = (
                            f"Error: Tool {tool_name} execution failed. Details: {str(e)}\n"
                            "请调整工具的参数，重新执行"
                        )

                    yield StreamToolCallResponse(
                        tool_call_end=ToolCallEnd(
                            tool_call_id=tool_call_id,
                            tool_name=tool_name,
                            tool_arguments=parsed_arguments,
                            tool_result=str(tool_result),
                        )
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "content": str(tool_result),  # 确保调用结果为字符串
                            "tool_call_id": tool_call_id,
                        }
                    )

                # 带着工具结果进入下一轮循环


def main():
    messages = [{"role": "user", "content": "你好，介绍一下你自己吧！"}]
    client = OpenAIClient()
    for chunk in client.send_messages(messages, "deepseek"):
        print(chunk, end="")
    logger.info(client.token_usage)


if __name__ == "__main__":
    main()
