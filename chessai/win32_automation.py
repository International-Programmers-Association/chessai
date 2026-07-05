from __future__ import annotations

import ctypes
import ctypes.wintypes
import time
from typing import Callable, Optional

import chess

from chessai.screen_reader import BoardRegion

MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
WINDOW_BUFFER_SIZE = 512

WNDENUMPROC = ctypes.WINFUNCTYPE(
    ctypes.c_bool,
    ctypes.wintypes.HWND,
    ctypes.wintypes.LPARAM,
)

_user32 = ctypes.windll.user32

# ── Global hotkey via keyboard library ─────────────────────────────

def _mod_vk_to_str(mod_keys: int, vk: int) -> str:
    parts = []
    if mod_keys & 2:
        parts.append("ctrl")
    if mod_keys & 4:
        parts.append("shift")
    if mod_keys & 1:
        parts.append("alt")
    if mod_keys & 8:
        parts.append("win")

    # VK to keyboard-library key name
    if 0x30 <= vk <= 0x39:
        k = chr(ord("0") + vk - 0x30)
    elif 0x41 <= vk <= 0x5A:
        k = chr(ord("a") + vk - 0x41)
    elif 0x70 <= vk <= 0x7B:
        k = f"f{vk - 0x70 + 1}"
    elif vk == 0x08:
        k = "backspace"
    elif vk == 0x09:
        k = "tab"
    elif vk == 0x0D:
        k = "enter"
    elif vk == 0x1B:
        k = "escape"
    elif vk == 0x20:
        k = "space"
    elif vk == 0x24:
        k = "home"
    elif vk == 0x23:
        k = "end"
    elif vk == 0x21:
        k = "page up"
    elif vk == 0x22:
        k = "page down"
    elif vk == 0x2E:
        k = "delete"
    elif vk == 0x25:
        k = "left"
    elif vk == 0x26:
        k = "up"
    elif vk == 0x27:
        k = "right"
    elif vk == 0x28:
        k = "down"
    elif 0x60 <= vk <= 0x69:
        k = f"num {vk - 0x60}"
    elif vk == 0x6A:
        k = "multiply"
    elif vk == 0x6B:
        k = "add"
    elif vk == 0x6D:
        k = "subtract"
    elif vk == 0x6E:
        k = "decimal"
    elif vk == 0x6F:
        k = "divide"
    elif vk == 0xBA:
        k = ";"
    elif vk == 0xBB:
        k = "+"
    elif vk == 0xBC:
        k = ","
    elif vk == 0xBD:
        k = "-"
    elif vk == 0xBE:
        k = "."
    elif vk == 0xBF:
        k = "/"
    elif vk == 0xC0:
        k = "`"
    elif vk == 0xDB:
        k = "["
    elif vk == 0xDC:
        k = "\\"
    elif vk == 0xDD:
        k = "]"
    elif vk == 0xDE:
        k = "'"
    else:
        k = f"vk{vk:x}"

    parts.append(k)
    return "+".join(parts)


def install_hotkey(
    callback: Callable[[], None],
    *,
    mod_keys: int = 3,
    vk: int = 0x53,
) -> None:
    """Register a global hotkey via ``keyboard.add_hotkey``.

    The *callback* is called from a background thread — use
    ``root.after(0, cb)`` to dispatch to the GUI thread.
    """
    import keyboard
    uninstall_hotkey()
    hotkey_str = _mod_vk_to_str(mod_keys, vk)
    keyboard.add_hotkey(hotkey_str, callback, suppress=True)


def uninstall_hotkey() -> None:
    """Unregister all hotkeys."""
    import keyboard
    try:
        keyboard.clear_all_hotkeys()
    except Exception:
        pass


def _enum_window_callback(hwnd: int, lparam: int) -> bool:
    arr = ctypes.cast(lparam, ctypes.POINTER(ctypes.py_object)).contents.value
    if _user32.IsWindowVisible(hwnd):
        arr.append(hwnd)
    return True


def _get_window_title(hwnd: int) -> str:
    buf = ctypes.create_unicode_buffer(WINDOW_BUFFER_SIZE)
    _user32.GetWindowTextW(hwnd, buf, WINDOW_BUFFER_SIZE)
    return buf.value


def find_window(title_pattern: str) -> Optional[int]:
    hwnds: list[int] = []
    callback = WNDENUMPROC(_enum_window_callback)
    ptr = ctypes.pointer(ctypes.py_object(hwnds))
    _user32.EnumWindows(callback, ctypes.cast(ptr, ctypes.wintypes.LPARAM))
    title_lower = title_pattern.lower()
    for hwnd in hwnds:
        if title_lower in _get_window_title(hwnd).lower():
            return hwnd
    return None


def activate_window(hwnd: int) -> None:
    _user32.ShowWindow(hwnd, 9)
    _user32.SetForegroundWindow(hwnd)


def chess_square_to_screen(
    square: chess.Square,
    region: BoardRegion,
    flipped: bool,
) -> tuple[int, int]:
    file_i = chess.square_file(square)
    rank_i = chess.square_rank(square)
    sq_size = region.width / 8.0

    if flipped:
        x = region.left + file_i * sq_size + sq_size / 2
        y = region.top + rank_i * sq_size + sq_size / 2
    else:
        x = region.left + file_i * sq_size + sq_size / 2
        y = region.top + (7 - rank_i) * sq_size + sq_size / 2

    return (int(x), int(y))


def click_at(x: int, y: int) -> None:
    _user32.SetCursorPos(x, y)
    _user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    time.sleep(0.003)
    _user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


def make_move(
    move: chess.Move,
    region: BoardRegion,
    flipped: bool,
    *,
    hwnd: Optional[int] = None,
    click_delay: float = 0.02,
) -> None:
    if hwnd is not None:
        activate_window(hwnd)
        time.sleep(0.05)

    from_x, from_y = chess_square_to_screen(move.from_square, region, flipped)
    click_at(from_x, from_y)
    time.sleep(click_delay)

    to_x, to_y = chess_square_to_screen(move.to_square, region, flipped)
    click_at(to_x, to_y)


def get_foreground_window() -> Optional[int]:
    hwnd = _user32.GetForegroundWindow()
    return hwnd if hwnd else None


def get_foreground_window_title() -> str:
    hwnd = get_foreground_window()
    if hwnd:
        return _get_window_title(hwnd)
    return ""
