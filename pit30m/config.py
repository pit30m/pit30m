import os
from typing import Optional

PIT30M_CACHE_DIR_ENV_VAR = "PIT30M_CACHE_DIR"

def get_pit30m_cache_dir() -> Optional[str]:
    """Returns the directory where Pit30M caches various things.

    If this is unset, we use a default cache path. If if is set to an empty string or "disable", we disable caching.
    """
    env_dir = os.getenv(PIT30M_CACHE_DIR_ENV_VAR)
    default_cache_dir = os.path.expanduser("~/.cache/pit30m")

    if env_dir is None:
        return default_cache_dir
    elif env_dir == "" or env_dir.lower() == "disable":
        return None
    else:
        return env_dir
