from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping
from urllib.parse import urlparse


class ModelError(ValueError):
    """A model profile is malformed or its credentials are unavailable."""


@dataclass(frozen=True)
class ModelProfile:
    name: str
    base_url: str | None = None
    api_key_env: str = "DEEPSEEK_API_KEY"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ModelError(f"name must be a non-empty string: {self.name!r}")
        if not isinstance(self.api_key_env, str) or not self.api_key_env.strip():
            raise ModelError(
                f"api_key_env must be a non-empty string: {self.api_key_env!r}"
            )
        if self.base_url is not None:
            if not isinstance(self.base_url, str):
                raise ModelError(f"base_url must be a string: {self.base_url!r}")
            parsed = urlparse(self.base_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ModelError(
                    f"base_url must be an http(s) URL: {self.base_url!r}"
                )

    def api_key(self, env: Mapping[str, str] | None = None) -> str:
        source = os.environ if env is None else env
        value = source.get(self.api_key_env, "")
        if not value.strip():
            raise ModelError(f"{self.api_key_env} is not set")
        return value

    def redacted(self) -> dict[str, object]:
        return {
            "name": self.name,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
        }


DEEPSEEK_FLASH = ModelProfile(
    name="deepseek-v4-flash",
    base_url="https://api.deepseek.com/v1",
    api_key_env="DEEPSEEK_API_KEY",
)
