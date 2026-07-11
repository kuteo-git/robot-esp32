from core.utils.llm import create_instance
from core.handle.clientConfig import client_allows, client_allowlist, host_allowed

TAG = __name__

# Only OpenAI-compatible providers are supported for per-session override this release.
# Other transports (e.g. anthropic / OmniRoute) fall back to the global LLM — documented follow-up.
ALLOWED_TRANSPORTS = {"openai"}


def build_client_llm(llm_config, server_config, logger=None):
    """Build a per-session LLM from a client-supplied config, or None to fall back to the global LLM.

    Returns (instance_or_None, reason). See the plan/spec for the reason vocabulary.
    """
    if not client_allows(server_config, "llm"):
        return None, "disabled"
    if not isinstance(llm_config, dict):
        return None, "no_config"

    base_url = llm_config.get("base_url")
    model_name = llm_config.get("model_name")
    if not base_url or not model_name:
        return None, "missing_fields"

    transport = str(llm_config.get("type") or llm_config.get("transport") or "openai").lower()
    if transport not in ALLOWED_TRANSPORTS:
        return None, f"bad_transport:{transport}"

    if not host_allowed(base_url, client_allowlist(server_config)):
        return None, "host_not_allowed"

    provider_conf = {
        "type": transport,
        "base_url": base_url,
        "model_name": model_name,
        "api_key": llm_config.get("api_key", ""),
    }
    try:
        instance = create_instance(transport, provider_conf)
        return instance, "ok"
    except Exception as e:  # provider __init__ or import failure -> fall back, don't crash the session
        if logger is not None:
            logger.bind(tag=TAG).warning(f"client llm build failed: {e}")
        return None, f"build_failed:{e}"
