import os

import pytest

from evo.adapters.deepseek import DEEPSEEK_FLASH, ModelError, ModelProfile


def test_flash_profile_targets_deepseek():
    assert DEEPSEEK_FLASH.name == "deepseek-v4-flash"
    assert DEEPSEEK_FLASH.base_url == "https://api.deepseek.com/v1"
    assert DEEPSEEK_FLASH.api_key_env == "DEEPSEEK_API_KEY"


def test_api_key_reads_the_named_variable(monkeypatch):
    monkeypatch.setenv("EVO_TEST_KEY", "secret-value")
    profile = ModelProfile(name="m", api_key_env="EVO_TEST_KEY")
    assert profile.api_key() == "secret-value"


def test_api_key_error_names_the_variable_without_its_value(monkeypatch):
    monkeypatch.delenv("EVO_TEST_KEY", raising=False)
    profile = ModelProfile(name="m", api_key_env="EVO_TEST_KEY")
    with pytest.raises(ModelError) as excinfo:
        profile.api_key()
    assert "EVO_TEST_KEY" in str(excinfo.value)


def test_redacted_never_contains_the_key(monkeypatch):
    monkeypatch.setenv("EVO_TEST_KEY", "secret-value")
    profile = ModelProfile(name="m", api_key_env="EVO_TEST_KEY")
    payload = profile.redacted()
    assert payload == {"name": "m", "base_url": None, "api_key_env": "EVO_TEST_KEY"}
    assert "secret-value" not in str(payload)


def test_rejects_empty_name():
    with pytest.raises(ModelError, match="name"):
        ModelProfile(name="")


def test_rejects_empty_api_key_env():
    with pytest.raises(ModelError, match="api_key_env"):
        ModelProfile(name="m", api_key_env="")


def test_rejects_base_url_without_http_scheme():
    with pytest.raises(ModelError, match="base_url"):
        ModelProfile(name="m", base_url="api.deepseek.com/v1")


def test_repr_hides_the_key(monkeypatch):
    monkeypatch.setenv("EVO_TEST_KEY", "secret-value")
    profile = ModelProfile(name="m", api_key_env="EVO_TEST_KEY")
    assert "secret-value" not in repr(profile)
