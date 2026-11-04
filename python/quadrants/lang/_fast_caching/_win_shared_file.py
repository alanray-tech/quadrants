"""File operations that let processes sharing a cache directory read, replace and touch the same entry at once on
Windows.

A file opened with the builtin ``open`` (or touched with ``os.utime``) is not shared for deletion, so ``os.replace`` onto
it from another process fails with ``WinError 5``, and opening a file while it is being replaced fails with ``Errno 13``.
Here, readers and touches open the entry with ``FILE_SHARE_DELETE``, and replaces rename with POSIX semantics, which
swaps the name even while other handles are open: those keep reading the previous content.
"""

import sys

assert sys.platform == "win32"

import ctypes  # noqa: E402
import msvcrt  # noqa: E402
import os  # noqa: E402
from ctypes import wintypes  # noqa: E402

_GENERIC_READ = 0x80000000
_DELETE = 0x00010000
_FILE_WRITE_ATTRIBUTES = 0x00000100
_FILE_SHARE_ALL = 0x00000007  # read | write | delete
_OPEN_EXISTING = 3
_FILE_ATTRIBUTE_NORMAL = 0x80
_INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
_FILE_RENAME_INFO_EX = 22
_FILE_RENAME_FLAG_REPLACE_IF_EXISTS = 0x1
_FILE_RENAME_FLAG_POSIX_SEMANTICS = 0x2

_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_kernel32.CreateFileW.restype = ctypes.c_void_p
_kernel32.CreateFileW.argtypes = [
    wintypes.LPCWSTR,
    wintypes.DWORD,
    wintypes.DWORD,
    ctypes.c_void_p,
    wintypes.DWORD,
    wintypes.DWORD,
    ctypes.c_void_p,
]
_kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
_kernel32.SetFileInformationByHandle.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
_kernel32.GetSystemTimeAsFileTime.argtypes = [ctypes.POINTER(wintypes.FILETIME)]
_kernel32.SetFileTime.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(wintypes.FILETIME),
    ctypes.POINTER(wintypes.FILETIME),
    ctypes.POINTER(wintypes.FILETIME),
]


def _open_handle(path: str, access: int) -> int:
    handle = _kernel32.CreateFileW(
        os.path.abspath(path), access, _FILE_SHARE_ALL, None, _OPEN_EXISTING, _FILE_ATTRIBUTE_NORMAL, None
    )
    if handle == _INVALID_HANDLE_VALUE:
        err = ctypes.get_last_error()
        raise OSError(None, ctypes.FormatError(err), path, err)
    return handle


def read_text(path: str) -> str:
    """Read *path* as text in the default encoding, like the builtin ``open``, through a handle shared for reading,
    writing and deletion."""
    fd = msvcrt.open_osfhandle(_open_handle(path, _GENERIC_READ), os.O_RDONLY)
    with open(fd) as f:
        return f.read()


def replace(src: str, dst: str) -> None:
    """Rename *src* onto *dst* with POSIX semantics, which succeeds while other processes have *dst* open. Falls back to
    ``os.replace`` on file systems without POSIX rename support."""
    handle = _open_handle(src, _DELETE)
    try:
        name = os.path.abspath(dst)

        class _FileRenameInfo(ctypes.Structure):
            _fields_ = [
                ("Flags", wintypes.DWORD),
                ("RootDirectory", ctypes.c_void_p),
                ("FileNameLength", wintypes.DWORD),
                ("FileName", ctypes.c_wchar * (len(name) + 1)),
            ]

        info = _FileRenameInfo(
            _FILE_RENAME_FLAG_REPLACE_IF_EXISTS | _FILE_RENAME_FLAG_POSIX_SEMANTICS, None, len(name) * 2, name
        )
        renamed = _kernel32.SetFileInformationByHandle(
            handle, _FILE_RENAME_INFO_EX, ctypes.byref(info), ctypes.sizeof(info)
        )
    finally:
        _kernel32.CloseHandle(handle)
    if not renamed:
        os.replace(src, dst)


def touch(path: str) -> None:
    """Set the access and modification times of *path* to now, through a handle shared for deletion."""
    handle = _open_handle(path, _FILE_WRITE_ATTRIBUTES)
    try:
        now = wintypes.FILETIME()
        _kernel32.GetSystemTimeAsFileTime(ctypes.byref(now))
        if not _kernel32.SetFileTime(handle, None, ctypes.byref(now), ctypes.byref(now)):
            err = ctypes.get_last_error()
            raise OSError(None, ctypes.FormatError(err), path, err)
    finally:
        _kernel32.CloseHandle(handle)
