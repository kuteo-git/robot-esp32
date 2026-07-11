from urllib.parse import urlparse


def _cc(server_config):
    return server_config.get("client_config") or {}


def client_allows(server_config, feature):
    """True if client-BYO for `feature` is allowed: master `enabled` AND `allow_<feature>`.
    Missing keys default to True (permissive for this project; operators set them false)."""
    cc = _cc(server_config)
    return bool(cc.get("enabled", True) and cc.get(f"allow_{feature}", True))


def client_allowlist(server_config):
    return _cc(server_config).get("allowlist") or []


def host_allowed(base_url, allowlist):
    if not allowlist:
        return True
    host = (urlparse(base_url).hostname or "").lower()
    for d in allowlist:
        d = (d or "").lower().strip()
        if d and (host == d or host.endswith("." + d)):
            return True
    return False
