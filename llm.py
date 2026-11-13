import os
from dataclasses import dataclass

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from config import (
    API_KEY_VARS,
    DEFAULT_MODELS,
    DEFAULT_PROVIDER,
    MAX_TOKENS,
    REQUEST_TIMEOUT_S,
    SDK_RETRIES_DISABLED,
)


class ConfigError(Exception): ...


@dataclass(frozen=True)
class RoleModels:
    supervisor: BaseChatModel
    researcher: BaseChatModel
    analyst: BaseChatModel
    synthesizer: BaseChatModel


def selected_provider() -> str:
    provider = os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER).strip().lower()
    if provider not in DEFAULT_MODELS:
        raise ConfigError(
            f"Proveedor desconocido: {provider!r}. Opciones: {', '.join(DEFAULT_MODELS)}."
        )
    return provider


def model_name(provider: str, role: str) -> str:
    return os.getenv(f"{role.upper()}_MODEL") or DEFAULT_MODELS[provider][role]


def build_chat_model(provider: str, name: str) -> BaseChatModel:
    if provider == "ollama":
        return ChatOllama(
            model=name,
            temperature=0,
            num_predict=MAX_TOKENS,
            base_url=os.getenv("OLLAMA_BASE_URL") or None,
        )
    key_var = API_KEY_VARS[provider]
    api_key = os.getenv(key_var)
    if not api_key:
        raise ConfigError(f"Falta la variable de entorno {key_var}. Revisa tu archivo .env.")
    if provider == "openai":
        return ChatOpenAI(
            model=name,
            temperature=0,
            max_tokens=MAX_TOKENS,
            timeout=REQUEST_TIMEOUT_S,
            max_retries=SDK_RETRIES_DISABLED,
            api_key=api_key,
        )
    return ChatAnthropic(
        model=name,
        max_tokens=MAX_TOKENS,
        timeout=REQUEST_TIMEOUT_S,
        max_retries=SDK_RETRIES_DISABLED,
        api_key=api_key,
    )


def build_models() -> RoleModels:
    provider = selected_provider()
    supervisor = build_chat_model(provider, model_name(provider, "supervisor"))
    worker = build_chat_model(provider, model_name(provider, "worker"))
    return RoleModels(supervisor=supervisor, researcher=worker, analyst=worker, synthesizer=worker)
