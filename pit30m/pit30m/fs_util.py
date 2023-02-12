import os
from typing import Optional
from urllib.parse import urlparse

import fsspec
from joblib import Memory


memory = Memory(location=os.path.expanduser("~/.cache/pit30m"), verbose=0)

@memory.cache(ignore=["fs"], verbose=0)
def cached_glob_images(root_dir: str, fs: Optional[fsspec.AbstractFileSystem]) -> list[str]:
    scheme = urlparse(root_dir).scheme
    if fs is None:
        fs = fsspec.filesystem(scheme)
    entries = sorted(fs.glob(os.path.join(root_dir, "*", "*.webp")))
    if scheme is None or 0 == len(scheme):
        return entries
    else:
        return [f"{scheme}://{entry}" for entry in entries]



@memory.cache(ignore=["fs"], verbose=0)
def cached_glob_lidar_sweeps(root_dir: str, fs: Optional[fsspec.AbstractFileSystem]) -> list[str]:
    scheme = urlparse(root_dir).scheme
    if fs is None:
        fs = fsspec.filesystem(scheme)
    entries = sorted(fs.glob(os.path.join(root_dir, "*", "*.npz.lz4")))
    if scheme is None or 0 == len(scheme):
        return entries
    else:
        return [f"{scheme}://{entry}" for entry in entries]
