import json
from threading import Lock
from typing import AsyncGenerator, Callable, Generator, Optional

from openai import OpenAI, AsyncOpenAI
from openai.types import CompletionUsage, Completion

from llm_client.utils import log_helper
from llm_client.utils.data_processing import str_to_json
from llm_client.utils.llm_config import get_llm_config

logger = log_helper.get_logger()


class OpenAIClient:
    def __init__(self):
        """初始化OpenAI客户端，设置token使用统计"""
        self.token_usage_total = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
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

    def update_total_token_usage(self, usage: CompletionUsage) -> None:
        """更新累计的token使用量, 线程安全"""
        one_million = 1_000_000
        with self.lock:
            self.token_usage_total["prompt_tokens"] += usage.prompt_tokens
            self.token_usage_total["completion_tokens"] += usage.completion_tokens
            self.token_usage_total["total_tokens"] += usage.total_tokens
            prompt_tokens = self.token_usage_total["prompt_tokens"]
            completion_tokens = self.token_usage_total["completion_tokens"]
            cost = (
                prompt_tokens * self.input_cost + completion_tokens * self.output_cost
            ) / one_million
            self.cost = cost

    def _process_response(self, response: Completion) -> str:
        usage = response.usage
        # 累计token消耗总量
        if usage:
            logger.info(usage)
            self.update_total_token_usage(usage)
        return response.choices[0].message.content

    def send_messages(
        self, messages: list[dict], model: Optional[str] = "deepseek"
    ) -> str:
        """
        发送消息到大模型，并返回响应
        Args:
            messages: 消息列表
            model: 模型名称，默认为"deepseek"
        Return:
            纯文本响应内容
        """
        api_base, api_key, model_name = self._get_client_config(model)
        client = OpenAI(api_key=api_key, base_url=api_base)

        response = client.chat.completions.create(
            model=model_name, messages=messages, stream=False
        )

        return self._process_response(response)

    def get_json_response(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        extra_body: Optional[dict] = None,
    ) -> dict | list[dict] | list:
        """
        发送消息到大模型，并返回json格式的响应
        Args:
            messages: 消息列表
            model: 模型名称，默认为"deepseek"
        Return:
            json.loads()后的响应内容
        """
        api_base, api_key, model_name = self._get_client_config(model)
        client = OpenAI(api_key=api_key, base_url=api_base)

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=False,
            response_format={"type": "json_object"},
            extra_body=extra_body,
        )

        content = self._process_response(response)

        return str_to_json(content)

    async def get_json_response_async(
        self, messages: list[dict], model: Optional[str] = "deepseek"
    ) -> dict | list[dict] | list:
        """
        协程发送消息到大模型，并返回json格式的响应
        Args:
            messages: 消息列表
        Return:
            json.loads()后的响应内容
        """
        api_base, api_key, model_name = self._get_client_config(model)
        async with AsyncOpenAI(api_key=api_key, base_url=api_base) as client:
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=False,
                response_format={"type": "json_object"},
            )

        content = self._process_response(response)

        return str_to_json(content)

    def send_messages_stream(
        self,
        messages: list[dict],
        config_name: Optional[str] = "deepseek",
        response_format: Optional[dict] = None,
        stop: Optional[list[str]] = None,
    ) -> Generator[str, None, str]:
        """
        发送消息到大模型，并返回流式响应，处理内容过长导致的截断
        Args:
            messages: 消息列表
            config_name: 配置名称，默认为'deepseek'
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
                )

                for chunk in response:
                    if hasattr(chunk, "usage") and chunk.usage:  # 检查是否有 usage 信息
                        logger.info(chunk.usage)
                        usage = chunk.usage
                        self.update_total_token_usage(usage)
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
                logger.error(f"Error during streaming: {e}")
                break

        return full_response

    def send_messages_stream_dict(
        self,
        messages: list[dict],
        config_name: Optional[str] = "deepseek",
        response_format: Optional[dict] = None,
    ) -> Generator[dict, None, dict]:
        """
        发送消息到大模型，并以dict类型返回流式响应，处理内容过长导致的截断
        Args:
            messages: 消息列表
            config_name: 配置名称，默认为'deepseek'
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
            )

            for chunk in response:
                if hasattr(chunk, "usage") and chunk.usage:  # 检查是否有 usage 信息
                    logger.info(chunk.usage)
                    usage = chunk.usage
                    self.update_total_token_usage(usage)
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
        stop: list[str] = None,
        tool_argument_to_show: list[str] = (),
        parallel_tool_calls: bool = True,
    ) -> AsyncGenerator[dict, dict]:
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
            tool_argument_to_show: 会yield指定参数的值，元素为参数名
        """
        api_base, api_key, model_name = self._get_client_config(config_name)

        async with AsyncOpenAI(api_key=api_key, base_url=api_base) as client:
            content_all = ""
            messages = messages.copy()  # 防止改变原变量

            while True:
                logger.info(
                    f"Sending messages to LLM. Current message count: {len(messages)}"
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
                )

                current_round_content = ""
                # 使用字典来重构工具调用，以其索引为键
                tool_call_deltas_by_index = {}

                async for chunk in response:
                    # logger.info(chunk)

                    if hasattr(chunk, "usage") and chunk.usage:  # 检查是否有 usage 信息
                        logger.info(chunk.usage)
                        usage = chunk.usage
                        self.update_total_token_usage(usage)
                    else:  # 如果没有 usage，则处理 choices 内容
                        delta = chunk.choices[0].delta
                        if hasattr(delta, "reasoning_content"):
                            yield {"reasoning_content": delta.reasoning_content}

                        if delta.content:
                            current_round_content += delta.content
                            yield {"content": delta.content}
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
                                        yield {
                                            "tool_call": f"\n调用工具{tool_call_delta.function.name}\n\n"
                                        }
                                        tool_call_deltas_by_index[index]["function"][
                                            "name"
                                        ] = tool_call_delta.function.name
                                    if tool_call_delta.function.arguments:
                                        yield {
                                            "tool_call": f"{tool_call_delta.function.arguments}"
                                        }
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
                logger.info(f"Assistant turn content: {current_round_content}")
                if assistant_tool_calls_reconstructed:
                    logger.info(
                        f"Assistant turn tool calls: "
                        f"{json.dumps(assistant_tool_calls_reconstructed, indent=2, ensure_ascii=False)}"
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
                    logger.warning(
                        "LLM response was empty for this turn. "
                        "Returning accumulated content if any, or empty string."
                    )
                    content_all += current_round_content
                    # return content_all
                    return

                # 没有进一步的工具调用，说明回答结束
                if not has_tool_calls:
                    logger.info(
                        "LLM processing finished. No more tool calls requested."
                    )
                    content_all += current_round_content
                    # return content_all
                    return

                # 如果有工具调用，执行工具
                logger.info("LLM requested tool calls. Executing now.")
                yield {"tool_call": "\n\n正在执行相关工具\n\n"}
                for tool_call_obj in assistant_tool_calls_reconstructed:
                    tool_call_id = tool_call_obj["id"]
                    tool_name = tool_call_obj["function"]["name"]
                    arguments_str = tool_call_obj["function"]["arguments"]

                    # 检查工具参数是否完整，arguments_str可能为空
                    if not all([tool_call_id, tool_name]):
                        logger.error(
                            f"Malformed tool call object: ID or name missing. "
                            f"Skipping. Object: {tool_call_obj}"
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
                        yield {"tool_call": "工具调用失败，参数格式不正确"}
                        continue

                    logger.info(
                        f"Executing tool: {tool_name} (ID: {tool_call_id}) with arguments: {arguments_str}"
                    )

                    try:
                        parsed_arguments = json.loads(arguments_str)
                        for arg in tool_argument_to_show:
                            yield {arg: f"\n\n{parsed_arguments[arg]}\n\n"}
                        tool_result = await call_tool_func(tool_name, parsed_arguments)
                    except json.JSONDecodeError as e:
                        logger.error(
                            f"JSON decoding error for tool {tool_name} (ID: {tool_call_id}) "
                            f"arguments '{arguments_str}': {e}"
                        )
                        tool_result = (
                            f"Error: Tool {tool_name} arguments were not valid JSON. "
                            f"Error: {e}. Arguments received: {arguments_str}"
                        )
                    except Exception as e:
                        logger.error(
                            f"Error calling tool {tool_name} (ID: {tool_call_id}): {e}"
                        )
                        tool_result = (
                            f"Error: Tool {tool_name} execution failed. Details: {str(e)}\n"
                            "请调整工具的参数，重新执行"
                        )

                    yield {"tool_call": f"工具执行结果{str(tool_result)}"}
                    yield {
                        "tool_call_result": {
                            "tool_name": tool_name,
                            "result": str(tool_result),
                        }
                    }
                    yield {"content": "\n\n"}  # 工具执行成功后，插入两个换行符
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
    logger.info(client.token_usage_total)


if __name__ == "__main__":
    main()
