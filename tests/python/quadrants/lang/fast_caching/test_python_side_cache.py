import builtins
import os
import pathlib

import pytest

import quadrants as qd
from quadrants.lang._fast_caching.python_side_cache import PythonSideCache

from tests.helpers import test_utils
from tests.helpers.runtime import qd_init_same_arch


def _permission_denied(*args, **kwargs):
    raise PermissionError(13, "Permission denied")


@test_utils.test(arch=qd.cpu)
def test_python_side_cache_store_skips_entry_held_by_another_process(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qd_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)
    cache = PythonSideCache()
    cache.store("key", "old")

    # On Windows, `os.replace` onto an entry that another process has open fails with WinError 5
    monkeypatch.setattr(os, "replace", _permission_denied)
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

    # On Windows, opening an entry that another process is replacing fails with Errno 13
    monkeypatch.setattr(builtins, "open", _permission_denied)
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

    monkeypatch.setattr(os, "utime", _permission_denied)
    assert cache.try_load("key") == "value"
