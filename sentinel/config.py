from __future__ import annotations

import yaml
from pydantic import BaseModel, ValidationError


class SentinelConfigError(Exception):
    pass


class SentinelConfig(BaseModel):
    target_base_url: str
    api_key: str | None = None
    policy_id: str
    fixtures_dir: str
    agent_id: str = "sentinel"


def load_config(path: str) -> SentinelConfig:
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        raise SentinelConfigError(f"Config file not found: {path}")
    except yaml.YAMLError as e:
        raise SentinelConfigError(f"Invalid YAML in {path}: {e}")

    try:
        return SentinelConfig.model_validate(data)
    except ValidationError as e:
        raise SentinelConfigError(f"Config validation failed: {e}")
