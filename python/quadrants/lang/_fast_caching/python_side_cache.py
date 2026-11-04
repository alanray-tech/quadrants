import json
import os
import tempfile
import warnings

import pydantic

from .. import impl


class PythonSideCache:
    """
    Manages a cache that is managed from the python side (we also have c++-side caches)

    The cache is disk-based. When we create the PythonSideCache object, the cache
    path is created as a sub-folder of CompileConfig.offline_cache_file_path.

    Note that constructing this object is cheap, so there is no need to maintain some
    kind of conceptual singleton instance or similar.

    Each cache key value is stored to a single file, with the cache key as the filename.

    No metadata is associated with the file, making management very lightweight.

    We update the file date/time when we read from a particular file, so we can easily
    implement an LRU cleaning strategy at some point in the future, based on the file
    date/times.
    """

    def __init__(self) -> None:
        _cache_parent_folder = impl.get_runtime().prog.config().offline_cache_file_path
        self.cache_folder = os.path.join(_cache_parent_folder, "python_side_cache")
        os.makedirs(self.cache_folder, exist_ok=True)

    def _get_filepath(self, key: str) -> str:
        filepath = os.path.join(self.cache_folder, f"{key}.cache.txt")
        return filepath

    def _touch(self, filepath):
        """
        Updates file date/time.
        """
        try:
            with open(filepath, "a"):
                os.utime(filepath, None)
        except OSError:
            # Another process is replacing or removing the entry; the date only drives a future LRU cleanup
            pass

    def store(self, fast_cache_key: str, value: str) -> None:
        filepath = self._get_filepath(fast_cache_key)
        tmp_path = None

        target_dir = os.path.dirname(filepath)
        fd, tmp_path = tempfile.mkstemp(dir=target_dir, prefix=f"{fast_cache_key}.", suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            f.write(value)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.replace(tmp_path, filepath)
        except OSError:
            # Windows refuses to replace an entry that another process has open, and that process stores an
            # equivalent value: skip this store rather than fail the kernel call
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def try_load(self, fast_cache_key: str) -> str | None:
        filepath = self._get_filepath(fast_cache_key)
        if not os.path.isfile(filepath):
            return None
        try:
            with open(filepath) as f:
                res = f.read()
        except (OSError, pydantic.ValidationError, json.JSONDecodeError, UnicodeDecodeError) as e:
            # OSError: another process replaced or removed the entry meanwhile (Windows, or a stale NFS handle)
            warnings.warn(f"Failed to read from cache at {filepath} {e}")
            return None
        self._touch(filepath)
        return res
