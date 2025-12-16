from pathlib import Path
from typing import Optional, Any, Dict
from pydantic import BaseModel
import yaml

config_file_path = Path(__file__).parent.parent / "conf" / "openai_llms.yaml"


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
