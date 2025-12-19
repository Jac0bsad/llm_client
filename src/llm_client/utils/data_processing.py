"""
数据处理相关的工具函数
"""

import json


def str_to_json(content: str) -> dict | list[dict]:
    """
    将大模型输出的json格式字符串进行处理后加载为Python对象
    """
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    return json.loads(content)
