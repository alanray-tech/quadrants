import os
import pathlib
import sys

import pytest

import quadrants as qd
from quadrants.lang._fast_caching import python_side_cache
from quadrants.lang._fast_caching.python_side_cache import PythonSideCache

from tests.helpers import test_utils
from tests.helpers.runtime import qd_init_same_arch


def _permission_denied(*args, **kwargs):
    raise PermissionError(13, "Permission denied")


@test_utils.test(arch=qd.cpu)
def test_python_side_cache_store_skips_entry_that_cannot_be_replaced(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qd_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)
    cache = PythonSideCache()
    cache.store("key", "old")

    # Replacing an entry that another process has open fails with WinError 5 on Windows without POSIX rename
    monkeypatch.setattr(python_side_cache, "_replace", _permission_denied)
    cache.store("key", "new")
    monkeypatch.undo()

    assert cache.try_load("key") == "old"
    assert not list(pathlib.Path(cache.cache_folder).glob("*.tmp"))


@test_utils.test(arch=qd.cpu)
def test_python_side_cache_load_misses_entry_being_replaced(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qd_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)
    cache = PythonSideCache()
    cache.store("key", "value")

    # Opening an entry that another process is replacing or removing fails with Errno 13 on Windows
    monkeypatch.setattr(python_side_cache, "_read_text", _permission_denied)
    with pytest.warns(UserWarning, match="Failed to read from cache"):
        assert cache.try_load("key") is None
    monkeypatch.undo()

    assert cache.try_load("key") == "value"


@test_utils.test(arch=qd.cpu)
def test_python_side_cache_load_keeps_value_when_touch_fails(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qd_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)
    cache = PythonSideCache()
    cache.store("key", "value")

    monkeypatch.setattr(python_side_cache, "_touch_file", _permission_denied)
    assert cache.try_load("key") == "value"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows file sharing semantics")
@test_utils.test(arch=qd.cpu)
def test_python_side_cache_replaces_and_reads_entry_held_open_on_windows(tmp_path: pathlib.Path) -> None:
    """Another process reading an entry holds a handle on it, shared for deletion: storing a new value must replace the
    entry anyway, a load must read the new value, and the open handle keeps reading the previous one."""
    from quadrants.lang._fast_caching import _win_shared_file  # pylint: disable=C0415

    qd_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)
    cache = PythonSideCache()
    cache.store("key", "old")
    entry = os.path.join(cache.cache_folder, "key.cache.txt")

    import msvcrt  # pylint: disable=C0415

    fd = msvcrt.open_osfhandle(_win_shared_file._open_handle(entry, _win_shared_file._GENERIC_READ), os.O_RDONLY)
    with open(fd) as held:
        cache.store("key", "new")
        assert cache.try_load("key") == "new"
        assert held.read() == "old"
    assert not list(pathlib.Path(cache.cache_folder).glob("*.tmp"))
