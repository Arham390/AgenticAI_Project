import os
from importlib import import_module
from typing import Any, Optional


def _env_key(name: str) -> str:
    """Trim whitespace/newlines — common when pasting keys on Windows."""
    raw = os.getenv(name)
    return (raw or "").strip()


def normalize_llm_env_vars() -> None:
    """Write stripped values back so LangChain / Groq SDKs see clean keys."""
    for k in ("GROQ_API_KEY", "OPENAI_API_KEY", "GROQ_MODEL", "OPENAI_MODEL"):
        v = os.getenv(k)
        if v is not None and v.strip() != v:
            os.environ[k] = v.strip()


def _chat_groq_class():
    try:
        mod = import_module("langchain_groq")
        return getattr(mod, "ChatGroq", None)
    except Exception:
        return None


def _chat_openai_class():
    for module_name in ("langchain_openai", "langchain.chat_models"):
        try:
            module = import_module(module_name)
            return getattr(module, "ChatOpenAI", None)
        except Exception:
            continue
    return None


def get_chat_llm(*, temperature: float = 0.7, model: Optional[str] = None) -> Optional[Any]:
    """
    Prefer Groq when GROQ_API_KEY is set, else OpenAI when OPENAI_API_KEY is set.
    Model: GROQ_MODEL / OPENAI_MODEL env, or sensible defaults.
    """
    if _env_key("GROQ_API_KEY"):
        ChatGroq = _chat_groq_class()
        if ChatGroq:
            m = model or _env_key("GROQ_MODEL") or "llama-3.3-70b-versatile"
            try:
                return ChatGroq(model=m, temperature=temperature)
            except Exception:
                try:
                    return ChatGroq(temperature=temperature)
                except Exception:
                    pass

    if _env_key("OPENAI_API_KEY"):
        ChatOpenAI = _chat_openai_class()
        if ChatOpenAI:
            m = model or _env_key("OPENAI_MODEL") or "gpt-4o-mini"
            try:
                return ChatOpenAI(model=m, temperature=temperature)
            except Exception:
                try:
                    return ChatOpenAI(temperature=temperature)
                except Exception:
                    try:
                        return ChatOpenAI()
                    except Exception:
                        pass
    return None


def llm_configured() -> bool:
    return bool(_env_key("GROQ_API_KEY") or _env_key("OPENAI_API_KEY"))


def llm_provider_hint() -> str:
    if _env_key("GROQ_API_KEY"):
        return "groq"
    if _env_key("OPENAI_API_KEY"):
        return "openai"
    return "none"


def describe_llm(llm: Any) -> dict[str, str]:
    """Best-effort provider + model id from a bound LangChain chat model."""
    cls = llm.__class__.__name__
    if "Groq" in cls:
        provider = "groq"
    elif "OpenAI" in cls:
        provider = "openai"
    else:
        provider = cls.lower()
    model = (
        getattr(llm, "model_name", None)
        or getattr(llm, "model", None)
        or getattr(llm, "model_id", None)
        or "unknown"
    )
    return {"provider": provider, "model": str(model)}
