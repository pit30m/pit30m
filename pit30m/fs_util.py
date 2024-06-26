"""Filesystem utilities for Pit30M. Author: """

import os
from typing import Optional
from urllib.parse import urlparse

import fsspec
from joblib import Memory

from pit30m.config import get_pit30m_cache_dir

memory = Memory(location=get_pit30m_cache_dir(), verbose=0)


@memory.cache(ignore=["fs"])
def cached_glob(
    root_dir: str,
    extension: str,
    fs: Optional[fsspec.AbstractFileSystem] = None,
) -> list[str]:
    scheme = urlparse(root_dir).scheme
    storage_options = {}
    if scheme == "s3":
        storage_options["anon"] = True
    if fs is None:
        fs = fsspec.filesystem(scheme, **storage_options)
    entries = sorted(fs.glob(os.path.join(root_dir, "*", "*" + extension)))
    if scheme is None or 0 == len(scheme):
        return entries
    else:
        return [f"{scheme}://{entry}" for entry in entries]


def cached_glob_images(
    root_dir: str,
    fs: Optional[fsspec.AbstractFileSystem] = None,
) -> list[str]:
    return cached_glob(root_dir, ".webp", fs)


def cached_glob_lidar_sweeps(root_dir: str, fs: Optional[fsspec.AbstractFileSystem]) -> list[str]:
    return cached_glob(root_dir, ".npz.lz4", fs)
