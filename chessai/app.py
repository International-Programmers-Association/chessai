from __future__ import annotations

import os
import queue
import re
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import chess
import cv2

from chessai.board_widget import BoardWidget, MoveArrow, MY_ARROW, OPP_ARROW, ARROW_MY_COLOR, ARROW_OPP_COLOR, _color
from chessai.engine import AnalysisResult, StockfishEngine
from chessai.fast_matcher import match_backend_name
from chessai.screen_reader import BoardRegion, ScreenBoardReader, ScreenReadResult, select_screen_region
from chessai.win32_automation import install_hotkey, uninstall_hotkey

FEN_RE = re.compile(
    r"^([rnbqkpRNBQKP1-8]+/){7}[rnbqkpRNBQKP1-8]+ [wb] [KQkq-]+ [a-h1-8-]+ \d+ \d+$"
)

COLORS = {
    "bg": "#1e1e1e",
    "panel": "#262626",
    "text": "#e8e8e8",
    "muted": "#9a9a9a",
    "accent": "#81b64c",
    "eval_white": "#ffffff",
    "eval_black": "#1a1a1a",
}


class ChessAIApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("ChessAI — real-time analysis")
        self.root.configure(bg=COLORS["bg"])
        self.root.minsize(760, 520)

        self.engine = StockfishEngine()
        self.screen_reader = ScreenBoardReader()
        self.board = chess.Board()
        self.analysis_queue: queue.Queue[AnalysisResult] = queue.Queue()
        self.continuous = False
        self.screen_watch = False
        self.clipboard_watch = False
        self._last_clipboard = ""
        self._screen_job: Optional[str] = None
        self._clipboard_job: Optional[str] = None
        self.player_color: chess.Color = chess.WHITE
        self.auto_turn_var = tk.BooleanVar(value=True)
        self.turn_to_move_var = tk.StringVar(value="white")
        self.show_arrows_var = tk.BooleanVar(value=True)
        self._analysis_token = 0
        self._pending_analysis = False
        self.pinned_var = tk.BooleanVar(value=True)
        self._undo_stack: list[str] = []
        self.auto_play_var = tk.BooleanVar(value=False)
        self.move_delay_var = tk.DoubleVar(value=0.0)
        self.target_window_var = tk.StringVar(value="chess.com")
        self._pending_auto_move = False
        self._auto_play_hwnd: Optional[int] = None
        self._fen_history: list[str] = []

        self._load_config()
        self._save_config()

        self._build_ui()
        self._sync_turn_mode()
        self._set_always_on_top(True)
        self._start_engine()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        try:
            install_hotkey(
                lambda: self.root.after(0, self._toggle_auto_screen),
                mod_keys=self._hotkey_mod,
                vk=self._hotkey_vk,
            )
            self.root.after(0, self._update_hotkey_label)
            self.root.after(0, lambda: self.status_var.set(f"{self._hotkey_hint} to toggle screen capture"))
        except Exception:
            pass
        self._poll_analysis()

    def _build_ui(self) -> None:
        toolbar = tk.Frame(self.root, bg=COLORS["panel"], pady=4, padx=6)
        toolbar.pack(fill=tk.X)

        buttons = [
            ("New game", self._new_game),
            ("Undo move", self._undo),
            ("Load FEN", self._load_fen_dialog),
            ("Analyze", self._analyze_once),
            ("Analyze ∞", self._toggle_continuous),
            ("FEN buffer", self._toggle_clipboard_watch),
        ]
        for text, cmd in buttons:
            tk.Button(
                toolbar,
                text=text,
                command=cmd,
                bg="#3a3a3a",
                fg=COLORS["text"],
                activebackground="#4a4a4a",
                activeforeground=COLORS["text"],
                relief=tk.FLAT,
                padx=6,
                pady=2,
            ).pack(side=tk.LEFT, padx=2)

        self.auto_screen_btn = tk.Button(
            toolbar,
            text="▶ Auto from screen",
            command=self._toggle_auto_screen,
            bg=COLORS["accent"],
            fg="#ffffff",
            activebackground=COLORS["accent"],
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=8,
            pady=2,
        )
        self.auto_screen_btn.pack(side=tk.LEFT, padx=4)

        self.scan_btn = tk.Button(
            toolbar,
            text="Scan",
            command=self._scan_now,
            bg="#3a3a3a",
            fg=COLORS["text"],
            relief=tk.FLAT,
            padx=8,
            pady=2,
        )
        self.scan_btn.pack(side=tk.LEFT, padx=2)

        self.pin_btn = tk.Button(
            toolbar,
            text="📌 Always on top",
            command=self._toggle_pin,
            bg=COLORS["accent"],
            fg="#ffffff",
            activebackground=COLORS["accent"],
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=8,
            pady=2,
        )
        self.pin_btn.pack(side=tk.RIGHT, padx=2)

        self.reset_region_btn = tk.Button(
            toolbar,
            text="Reset region",
            command=self._reset_region,
            bg="#c0392b",
            fg="#ffffff",
            activebackground="#e74c3c",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=8,
            pady=2,
        )
        self.reset_region_btn.pack(side=tk.RIGHT, padx=2)
        self.pin_btn.pack(side=tk.RIGHT, padx=2)

        body = tk.Frame(self.root, bg=COLORS["bg"])
        body.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
        body.pack_propagate(False)

        left = tk.Frame(body, bg=COLORS["bg"])
        left.pack(side=tk.LEFT, fill=tk.Y)

        self.board_widget = BoardWidget(left, on_move=self._on_board_move)
        self.board_widget.pack()

        self.auto_mode_var = tk.StringVar(value="")
        tk.Label(
            left,
            textvariable=self.auto_mode_var,
            bg=COLORS["bg"],
            fg=COLORS["accent"],
            font=("Segoe UI", 8),
            wraplength=380,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(2, 0))
        tk.Label(
            left,
            bg=COLORS["bg"],
            fg=COLORS["muted"],
            font=("Segoe UI", 7),
            wraplength=380,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(1, 0))

        player_frame = tk.LabelFrame(
            left,
            text="Your color",
            bg=COLORS["bg"],
            fg=COLORS["muted"],
            font=("Segoe UI", 8),
            padx=6,
            pady=4,
        )
        player_frame.pack(fill=tk.X, pady=(4, 2))

        btn_row = tk.Frame(player_frame, bg=COLORS["bg"])
        btn_row.pack(fill=tk.X)

        self.white_btn = tk.Button(
            btn_row,
            text="♔ White",
            command=lambda: self._set_player_color(chess.WHITE),
            relief=tk.FLAT,
            padx=8,
            pady=4,
        )
        self.white_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))

        self.black_btn = tk.Button(
            btn_row,
            text="♚ Black",
            command=lambda: self._set_player_color(chess.BLACK),
            relief=tk.FLAT,
            padx=8,
            pady=4,
        )
        self.black_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)

        self.player_status_var = tk.StringVar()
        tk.Label(
            player_frame,
            textvariable=self.player_status_var,
            bg=COLORS["bg"],
            fg=COLORS["text"],
            font=("Segoe UI", 8),
        ).pack(anchor=tk.W, pady=(4, 0))

        legend = tk.Frame(player_frame, bg=COLORS["bg"])
        legend.pack(fill=tk.X, pady=(4, 0))
        tk.Label(legend, text="●", fg=_color(MY_ARROW, ARROW_MY_COLOR), bg=COLORS["bg"], font=("Segoe UI", 10)).pack(
            side=tk.LEFT
        )
        tk.Label(legend, text="your move", bg=COLORS["bg"], fg=COLORS["muted"], font=("Segoe UI", 8)).pack(
            side=tk.LEFT, padx=(2, 8)
        )
        tk.Label(legend, text="●", fg=_color(OPP_ARROW, ARROW_OPP_COLOR), bg=COLORS["bg"], font=("Segoe UI", 10)).pack(
            side=tk.LEFT
        )
        tk.Label(
            legend, text="opponent's move", bg=COLORS["bg"], fg=COLORS["muted"], font=("Segoe UI", 8)
        ).pack(side=tk.LEFT, padx=2)

        tk.Checkbutton(
            player_frame,
            text="Show arrows",
            variable=self.show_arrows_var,
            command=self._toggle_arrows,
            bg=COLORS["bg"],
            fg=COLORS["text"],
            selectcolor="#333333",
            activebackground=COLORS["bg"],
            activeforeground=COLORS["text"],
        ).pack(anchor=tk.W, pady=(4, 0))

        self._refresh_player_buttons()

        turn_frame = tk.LabelFrame(
            left,
            text="Who moves now",
            bg=COLORS["bg"],
            fg=COLORS["muted"],
            font=("Segoe UI", 8),
            padx=6,
            pady=4,
        )
        turn_frame.pack(fill=tk.X, pady=(2, 2))

        tk.Checkbutton(
            turn_frame,
            text="Auto (by position and moves)",
            variable=self.auto_turn_var,
            command=self._on_turn_mode_change,
            bg=COLORS["bg"],
            fg=COLORS["text"],
            selectcolor="#333333",
            activebackground=COLORS["bg"],
            activeforeground=COLORS["text"],
        ).pack(anchor=tk.W)

        turn_row = tk.Frame(turn_frame, bg=COLORS["bg"])
        turn_row.pack(fill=tk.X, pady=(4, 0))

        self.turn_white_btn = tk.Button(
            turn_row,
            text="♔ White's turn",
            command=lambda: self._set_turn_to_move(chess.WHITE),
            relief=tk.FLAT,
            padx=8,
            pady=3,
        )
        self.turn_white_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))

        self.turn_black_btn = tk.Button(
            turn_row,
            text="♚ Black's turn",
            command=lambda: self._set_turn_to_move(chess.BLACK),
            relief=tk.FLAT,
            padx=8,
            pady=3,
        )
        self.turn_black_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)

        self.turn_status_var = tk.StringVar()
        tk.Label(
            turn_frame,
            textvariable=self.turn_status_var,
            bg=COLORS["bg"],
            fg=COLORS["muted"],
            font=("Segoe UI", 8),
            wraplength=380,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(4, 0))

        self._refresh_turn_buttons()

        flip_btn = tk.Button(
            left,
            text="Flip board",
            command=self._flip_board,
            bg="#3a3a3a",
            fg=COLORS["text"],
            relief=tk.FLAT,
        )
        flip_btn.pack(pady=4)

        right = tk.Frame(body, bg=COLORS["panel"], width=340)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0))
        right.pack_propagate(False)

        right_canvas = tk.Canvas(right, bg=COLORS["panel"], highlightthickness=0, width=324)
        right_scrollbar = tk.Scrollbar(right, orient=tk.VERTICAL, command=right_canvas.yview)
        right_canvas.configure(yscrollcommand=right_scrollbar.set)
        right_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        right_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right_inner = tk.Frame(right_canvas, bg=COLORS["panel"])
        right_canvas.create_window((0, 0), window=right_inner, anchor=tk.NW,
                                   width=right_canvas.winfo_reqwidth())

        def _configure_right_inner(event):
            right_canvas.configure(scrollregion=right_canvas.bbox("all"))
        right_inner.bind("<Configure>", _configure_right_inner)

        def _on_right_canvas_configure(event):
            right_canvas.itemconfig(1, width=event.width)
        right_canvas.bind("<Configure>", _on_right_canvas_configure)

        def _on_right_mousewheel(event):
            right_canvas.yview_scroll(-1 * (event.delta // 120), "units")
        right_canvas.bind("<Enter>", lambda e: right_canvas.bind_all("<MouseWheel>", _on_right_mousewheel))
        right_canvas.bind("<Leave>", lambda e: right_canvas.unbind_all("<MouseWheel>"))

        tk.Label(
            right_inner,
            text="Analysis",
            font=("Segoe UI", 11, "bold"),
            bg=COLORS["panel"],
            fg=COLORS["text"],
        ).pack(anchor=tk.W, padx=8, pady=(8, 0))

        self.status_var = tk.StringVar(value="Engine starting…")
        tk.Label(
            right_inner,
            textvariable=self.status_var,
            bg=COLORS["panel"],
            fg=COLORS["muted"],
            font=("Segoe UI", 8),
        ).pack(anchor=tk.W, padx=8, pady=(0, 2))

        self.screen_status_var = tk.StringVar(value="Screen: inactive")
        tk.Label(
            right_inner,
            textvariable=self.screen_status_var,
            bg=COLORS["panel"],
            fg=COLORS["accent"],
            font=("Segoe UI", 8),
        ).pack(anchor=tk.W, padx=8, pady=(0, 4))

        eval_frame = tk.Frame(right_inner, bg=COLORS["panel"])
        eval_frame.pack(fill=tk.X, padx=8, pady=(0, 6))
        self.eval_canvas = tk.Canvas(
            eval_frame, width=20, height=160, highlightthickness=0, bg=COLORS["eval_black"]
        )
        self.eval_canvas.pack(side=tk.LEFT, fill=tk.Y)
        self.eval_label = tk.Label(
            eval_frame,
            text="0.00",
            font=("Consolas", 18, "bold"),
            bg=COLORS["panel"],
            fg=COLORS["text"],
        )
        self.eval_label.pack(side=tk.LEFT, padx=8)

        self.lines_frame = tk.Frame(right_inner, bg=COLORS["panel"])
        self.lines_frame.pack(fill=tk.X, padx=8)

        settings = tk.LabelFrame(
            right_inner,
            text="Settings",
            bg=COLORS["panel"],
            fg=COLORS["muted"],
            font=("Segoe UI", 8),
        )
        settings.pack(fill=tk.X, padx=6, pady=(6, 4))

        row = tk.Frame(settings, bg=COLORS["panel"])
        row.pack(fill=tk.X, padx=6, pady=3)
        tk.Label(row, text="Time (sec)", bg=COLORS["panel"], fg=COLORS["text"],
                 font=("Segoe UI", 8)).pack(side=tk.LEFT)
        self.time_var = tk.DoubleVar(value=0.05)
        ttk.Scale(row, from_=0.01, to=0.5, variable=self.time_var, orient=tk.HORIZONTAL).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=4
        )

        row_depth = tk.Frame(settings, bg=COLORS["panel"])
        row_depth.pack(fill=tk.X, padx=6, pady=3)
        tk.Label(row_depth, text="Depth", bg=COLORS["panel"], fg=COLORS["text"],
                 font=("Segoe UI", 8)).pack(side=tk.LEFT)
        self.depth_var = tk.IntVar(value=0)
        d_spin = ttk.Spinbox(row_depth, from_=0, to=30, textvariable=self.depth_var, width=4)
        d_spin.pack(side=tk.LEFT, padx=4)
        tk.Label(row_depth, text="(0 = no limit)", bg=COLORS["panel"], fg=COLORS["muted"],
                 font=("Segoe UI", 7)).pack(side=tk.LEFT)

        row2 = tk.Frame(settings, bg=COLORS["panel"])
        row2.pack(fill=tk.X, padx=6, pady=3)
        tk.Label(row2, text="Lines", bg=COLORS["panel"], fg=COLORS["text"],
                 font=("Segoe UI", 8)).pack(side=tk.LEFT)
        self.multipv_var = tk.IntVar(value=3)
        ttk.Spinbox(row2, from_=1, to=5, textvariable=self.multipv_var, width=4).pack(
            side=tk.LEFT, padx=4
        )

        row3 = tk.Frame(settings, bg=COLORS["panel"])
        row3.pack(fill=tk.X, padx=6, pady=3)
        tk.Label(row3, text="Threads", bg=COLORS["panel"], fg=COLORS["text"],
                 font=("Segoe UI", 8)).pack(side=tk.LEFT)
        self._threads_var = tk.IntVar(value=getattr(self, "_engine_threads", os.cpu_count() or 4))
        t_spin = ttk.Spinbox(row3, from_=1, to=256, textvariable=self._threads_var, width=4)
        t_spin.pack(side=tk.LEFT, padx=4)
        t_spin.bind("<<Increment>>", self._on_engine_config_change)
        t_spin.bind("<<Decrement>>", self._on_engine_config_change)
        tk.Label(row3, text="Hash", bg=COLORS["panel"], fg=COLORS["text"],
                 font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(8, 0))
        self._hash_var = tk.IntVar(value=getattr(self, "_engine_hash", 512))
        h_spin = ttk.Spinbox(row3, from_=16, to=65536, increment=64, textvariable=self._hash_var, width=5)
        h_spin.pack(side=tk.LEFT, padx=4)
        h_spin.bind("<<Increment>>", self._on_engine_config_change)
        h_spin.bind("<<Decrement>>", self._on_engine_config_change)

        row_buf = tk.Frame(settings, bg=COLORS["panel"])
        row_buf.pack(fill=tk.X, padx=6, pady=3)
        tk.Label(row_buf, text="Buffer", bg=COLORS["panel"], fg=COLORS["text"],
                 font=("Segoe UI", 8)).pack(side=tk.LEFT)
        self._history_size_var = tk.IntVar(value=100)
        b_spin = ttk.Spinbox(row_buf, from_=10, to=500, increment=10,
                             textvariable=self._history_size_var, width=4)
        b_spin.pack(side=tk.LEFT, padx=4)
        tk.Label(row_buf, text="(repetition buffer)", bg=COLORS["panel"], fg=COLORS["muted"],
                 font=("Segoe UI", 7)).pack(side=tk.LEFT)

        row_hk = tk.Frame(settings, bg=COLORS["panel"])
        row_hk.pack(fill=tk.X, padx=6, pady=3)
        tk.Label(row_hk, text="Hotkey", bg=COLORS["panel"], fg=COLORS["text"],
                 font=("Segoe UI", 8)).pack(side=tk.LEFT)
        self._hotkey_label = tk.Label(
            row_hk,
            text="",
            bg="#3a3a3a",
            fg=COLORS["accent"],
            font=("Segoe UI", 8, "bold"),
            padx=8,
            pady=2,
        )
        self._hotkey_label.pack(side=tk.LEFT, padx=4)
        tk.Button(
            row_hk,
            text="Change…",
            command=self._change_hotkey_dialog,
            bg="#3a3a3a",
            fg=COLORS["text"],
            relief=tk.FLAT,
            padx=6,
            pady=1,
            font=("Segoe UI", 7),
        ).pack(side=tk.LEFT)

        self._add_model_engine_selectors(settings)

        self.collect_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            settings,
            text="Collect data for AI",
            variable=self.collect_var,
            bg=COLORS["panel"],
            fg=COLORS["text"],
            selectcolor="#333333",
            activebackground=COLORS["panel"],
            activeforeground=COLORS["text"],
            font=("Segoe UI", 8),
        ).pack(anchor=tk.W, padx=6, pady=(0, 2))

        # ── Auto-play ──────────────────────────────────────────
        tk.Checkbutton(
            settings,
            text="Auto-play (best moves via WinAPI)",
            variable=self.auto_play_var,
            command=self._on_auto_play_toggle,
            bg=COLORS["panel"],
            fg=COLORS["text"],
            selectcolor="#333333",
            activebackground=COLORS["panel"],
            activeforeground=COLORS["text"],
            font=("Segoe UI", 8),
        ).pack(anchor=tk.W, padx=6, pady=(2, 0))

        ap_row = tk.Frame(settings, bg=COLORS["panel"])
        ap_row.pack(fill=tk.X, padx=6, pady=1)
        tk.Label(ap_row, text="Target:", bg=COLORS["panel"], fg=COLORS["text"],
                 font=("Segoe UI", 8)).pack(side=tk.LEFT)
        tk.Entry(
            ap_row,
            textvariable=self.target_window_var,
            bg="#333333",
            fg=COLORS["text"],
            relief=tk.FLAT,
            font=("Segoe UI", 8),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        tk.Button(
            ap_row,
            text="Detect",
            command=self._detect_target_window,
            bg="#3a3a3a",
            fg=COLORS["text"],
            relief=tk.FLAT,
            padx=4,
            font=("Segoe UI", 8),
        ).pack(side=tk.RIGHT)

        ap_row2 = tk.Frame(settings, bg=COLORS["panel"])
        ap_row2.pack(fill=tk.X, padx=6, pady=(0, 4))
        tk.Label(ap_row2, text="Delay:", bg=COLORS["panel"], fg=COLORS["text"],
                 font=("Segoe UI", 8)).pack(side=tk.LEFT)
        self._delay_scale = ttk.Scale(
            ap_row2, from_=0.0, to=1.0, variable=self.move_delay_var, orient=tk.HORIZONTAL
        )
        self._delay_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self._delay_val_label = tk.Label(
            ap_row2, text="0.0", bg=COLORS["panel"], fg=COLORS["text"], width=3,
            font=("Segoe UI", 8)
        )
        self._delay_val_label.pack(side=tk.LEFT)
        self.move_delay_var.trace_add("write", lambda *_: self._delay_val_label.config(
            text=f"{self.move_delay_var.get():.1f}"
        ))

        fen_frame = tk.Frame(self.root, bg=COLORS["panel"], padx=6, pady=4)
        fen_frame.pack(fill=tk.X)
        tk.Label(fen_frame, text="FEN", bg=COLORS["panel"], fg=COLORS["muted"],
                 font=("Segoe UI", 8)).pack(side=tk.LEFT)
        self.fen_var = tk.StringVar(value=self.board.fen())
        fen_entry = tk.Entry(
            fen_frame,
            textvariable=self.fen_var,
            bg="#333333",
            fg=COLORS["text"],
            insertbackground=COLORS["text"],
            relief=tk.FLAT,
            font=("Segoe UI", 8),
        )
        fen_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        fen_entry.bind("<Return>", lambda _e: self._apply_fen())
        tk.Button(
            fen_frame,
            text="Apply",
            command=self._apply_fen,
            bg=COLORS["accent"],
            fg="#ffffff",
            relief=tk.FLAT,
            padx=6,
            font=("Segoe UI", 8),
        ).pack(side=tk.LEFT)

    # ── Config ────────────────────────────────────────────────
    def _config_path(self) -> Path:
        return Path.home() / ".chessai" / "config.json"

    def _load_config(self) -> None:
        import json
        import chessai.classifier as clf
        from chessai import board_widget as bw
        from chessai.classifier import CURRENT_UNET_MODEL, MODELS_DIR
        from chessai.engine import DEFAULT_ENGINE_PATH

        cfg_path = self._config_path()
        if not cfg_path.exists():
            return
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
        except Exception:
            return

        md = cfg.get("models_dir", "")
        mf = cfg.get("model_file", "")
        if md and mf and Path(md, mf).exists():
            self._models_dir = md
            self._model_file = mf
            bw.CURRENT_PIECE_DIR = cfg.get("pieces_dir") or None

        ed = cfg.get("engines_dir", "")
        ef = cfg.get("engine_file", "")
        if ed and ef and Path(ed, ef).exists():
            self._engines_dir = ed
            self._engine_file = ef

        if cfg.get("light_color"):
            bw.BOARD_LIGHT = cfg["light_color"]
            self._light_color = cfg["light_color"]
        if cfg.get("dark_color"):
            bw.BOARD_DARK = cfg["dark_color"]
            self._dark_color = cfg["dark_color"]
        if cfg.get("arrow_my_color"):
            bw.ARROW_MY_COLOR = cfg["arrow_my_color"]
            self._arrow_my_color = cfg["arrow_my_color"]
        if cfg.get("arrow_opp_color"):
            bw.ARROW_OPP_COLOR = cfg["arrow_opp_color"]
            self._arrow_opp_color = cfg["arrow_opp_color"]

        torch_dir = cfg.get("torch_dir", "")
        if torch_dir:
            self._torch_dir = torch_dir

        self._engine_threads = cfg.get("engine_threads", os.cpu_count() or 4)
        self._engine_hash = cfg.get("engine_hash", 512)

        dev = cfg.get("device", "auto")
        self._device_choice = dev
        if dev != "auto":
            clf.FORCE_DEVICE = dev

        self._hotkey_mod = cfg.get("hotkey_mod", 3)  # MOD_CONTROL | MOD_ALT
        self._hotkey_vk = cfg.get("hotkey_vk", 0x53)  # VK_S
        self._hotkey_hint = self._format_hotkey(self._hotkey_mod, self._hotkey_vk)

    def _save_config(self) -> None:
        import json
        cfg_path = self._config_path()
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg = {
            "models_dir": getattr(self, "_models_dir", ""),
            "model_file": getattr(self, "_model_file", ""),
            "engines_dir": getattr(self, "_engines_dir", ""),
            "engine_file": getattr(self, "_engine_file", ""),
            "pieces_dir": getattr(self, "_pieces_dir", ""),
            "device": getattr(self, "_device_choice", "auto"),
            "light_color": getattr(self, "_light_color", ""),
            "dark_color": getattr(self, "_dark_color", ""),
            "arrow_my_color": getattr(self, "_arrow_my_color", ""),
            "arrow_opp_color": getattr(self, "_arrow_opp_color", ""),
            "torch_dir": getattr(self, "_torch_dir", ""),
            "engine_threads": getattr(self, "_engine_threads", os.cpu_count() or 4),
            "engine_hash": getattr(self, "_engine_hash", 512),
            "hotkey_mod": getattr(self, "_hotkey_mod", 3),
            "hotkey_vk": getattr(self, "_hotkey_vk", 0x53),
        }
        try:
            with open(cfg_path, "w") as f:
                json.dump(cfg, f, indent=2)
        except Exception:
            pass

    @staticmethod
    def _format_hotkey(mod: int, vk: int) -> str:
        parts = []
        if mod & 2:
            parts.append("Ctrl")
        if mod & 4:
            parts.append("Shift")
        if mod & 1:
            parts.append("Alt")
        if mod & 8:
            parts.append("Win")
        if 0x30 <= vk <= 0x39:
            k = chr(ord("0") + vk - 0x30)
        elif 0x41 <= vk <= 0x5A:
            k = chr(ord("A") + vk - 0x41)
        elif 0x70 <= vk <= 0x7B:
            k = f"F{vk - 0x70 + 1}"
        elif vk == 0x08:
            k = "Backspace"
        elif vk == 0x09:
            k = "Tab"
        elif vk == 0x0D:
            k = "Enter"
        elif vk == 0x1B:
            k = "Esc"
        elif vk == 0x20:
            k = "Space"
        elif vk == 0x24:
            k = "Home"
        elif vk == 0x23:
            k = "End"
        elif vk == 0x21:
            k = "Page Up"
        elif vk == 0x22:
            k = "Page Down"
        elif vk == 0x2E:
            k = "Delete"
        elif vk == 0x25:
            k = "Left"
        elif vk == 0x26:
            k = "Up"
        elif vk == 0x27:
            k = "Right"
        elif vk == 0x28:
            k = "Down"
        elif 0x60 <= vk <= 0x69:
            k = f"Numpad{vk - 0x60}"
        elif vk == 0x6A:
            k = "Num*"
        elif vk == 0x6B:
            k = "Num+"
        elif vk == 0x6D:
            k = "Num-"
        elif vk == 0x6E:
            k = "Num."
        elif vk == 0x6F:
            k = "Num/"
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
            k = f"VK_{vk:X}"
        parts.append(k)
        return "+".join(parts)

    def _update_hotkey_label(self) -> None:
        self._hotkey_label.config(text=self._hotkey_hint)

    def _change_hotkey_dialog(self) -> None:
        uninstall_hotkey()
        dialog = tk.Toplevel(self.root)
        dialog.title("Change hotkey")
        dialog.configure(bg=COLORS["bg"])
        dialog.resizable(False, False)
        dialog.grab_set()

        f = tk.Frame(dialog, bg=COLORS["bg"], padx=24, pady=20)
        f.pack()

        tk.Label(
            f, text="Press the new hotkey combination...",
            bg=COLORS["bg"], fg=COLORS["text"], font=("Segoe UI", 10),
        ).pack()

        display_var = tk.StringVar(value="…")
        display_lbl = tk.Label(
            f, textvariable=display_var,
            bg="#2a2a2a", fg=COLORS["accent"],
            font=("Segoe UI", 14, "bold"),
            padx=16, pady=8,
        )
        display_lbl.pack(pady=(10, 4))

        cancel_btn = tk.Button(
            f, text="Cancel", command=dialog.destroy,
            bg="#3a3a3a", fg=COLORS["text"], relief=tk.FLAT,
        )
        cancel_btn.pack(pady=(8, 0))

        pressed_mods: set = set()

        def on_key_press(event: tk.Event) -> None:
            nonlocal pressed_mods
            ks = event.keysym

            # Track modifiers
            if ks in ("Control_L", "Control_R"):
                pressed_mods.add(2)
                display_var.set("Ctrl + ?")
                return
            elif ks in ("Shift_L", "Shift_R"):
                pressed_mods.add(4)
                display_var.set("Shift + ?")
                return
            elif ks in ("Alt_L", "Alt_R", "Meta_L"):
                pressed_mods.add(1)
                display_var.set("Alt + ?")
                return
            elif ks in ("Super_L", "Super_R"):
                pressed_mods.add(8)
                display_var.set("Win + ?")
                return

            # Non-modifier key — capture the combo
            mods = sum(pressed_mods)
            vk = event.keycode
            name = key_name_from_keysym(ks, vk)

            # Save, reinstall, close
            uninstall_hotkey()
            self._hotkey_mod = mods
            self._hotkey_vk = vk
            self._hotkey_hint = self._format_hotkey(mods, vk)
            self._save_config()
            install_hotkey(
                lambda: self.root.after(0, self._toggle_auto_screen),
                mod_keys=mods, vk=vk,
            )
            self._update_hotkey_label()
            self.status_var.set(f"Hotkey changed to {self._hotkey_hint}")
            dialog.destroy()

        def key_name_from_keysym(keysym: str, keycode: int) -> str:
            m = {
                "F1": "F1", "F2": "F2", "F3": "F3", "F4": "F4",
                "F5": "F5", "F6": "F6", "F7": "F7", "F8": "F8",
                "F9": "F9", "F10": "F10", "F11": "F11", "F12": "F12",
            }
            if keysym in m:
                return m[keysym]
            if len(keysym) == 1:
                return keysym.upper()
            return f"VK_{keycode:X}"

        dialog.bind("<KeyPress>", on_key_press, "+")
        dialog.bind("<KeyRelease>", lambda e: pressed_mods.discard(
            2 if e.keysym in ("Control_L", "Control_R") else
            4 if e.keysym in ("Shift_L", "Shift_R") else
            1 if e.keysym in ("Alt_L", "Alt_R", "Meta_L") else
            8 if e.keysym in ("Super_L", "Super_R") else 0
        ), "+")

    # ── Model / Engine / Style selectors ──────────────────────
    def _add_model_engine_selectors(self, parent: tk.Frame) -> None:
        from chessai.classifier import MODELS_DIR as MDL_DIR
        from chessai.engine import DEFAULT_ENGINE_PATH
        from chessai.board_widget import CURRENT_PIECE_DIR

        self._models_dir = getattr(self, "_models_dir", str(MDL_DIR))
        self._engines_dir = getattr(self, "_engines_dir", str(DEFAULT_ENGINE_PATH.parent))

        # --- Model folder + file ---
        row_m_dir = tk.Frame(parent, bg=COLORS["panel"])
        row_m_dir.pack(fill=tk.X, padx=6, pady=(2, 0))
        tk.Label(row_m_dir, text="Models", bg=COLORS["panel"], fg=COLORS["text"],
                 font=("Segoe UI", 7)).pack(side=tk.LEFT)
        self._model_dir_var = tk.StringVar(value=self._models_dir)
        e_m = tk.Entry(row_m_dir, textvariable=self._model_dir_var, bg="#333", fg=COLORS["text"],
                       relief=tk.FLAT, font=("Segoe UI", 7))
        e_m.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 2))
        tk.Button(row_m_dir, text="Browse", command=self._pick_model_dir,
                  bg="#3a3a3a", fg=COLORS["text"], relief=tk.FLAT, padx=4, font=("Segoe UI", 7)).pack(side=tk.RIGHT)

        row_m_file = tk.Frame(parent, bg=COLORS["panel"])
        row_m_file.pack(fill=tk.X, padx=6, pady=(0, 1))
        self._model_file_var = tk.StringVar(value=getattr(self, "_model_file", ""))
        self._model_combo = ttk.Combobox(row_m_file, textvariable=self._model_file_var,
                                         state="readonly", width=20)
        self._model_combo.pack(side=tk.RIGHT)
        self._model_combo.bind("<<ComboboxSelected>>", self._on_model_change)
        self._refresh_model_combo()

        # --- Engine folder + file ---
        row_e_dir = tk.Frame(parent, bg=COLORS["panel"])
        row_e_dir.pack(fill=tk.X, padx=6, pady=(2, 0))
        tk.Label(row_e_dir, text="Engines", bg=COLORS["panel"], fg=COLORS["text"],
                 font=("Segoe UI", 7)).pack(side=tk.LEFT)
        self._engine_dir_var = tk.StringVar(value=self._engines_dir)
        e_e = tk.Entry(row_e_dir, textvariable=self._engine_dir_var, bg="#333", fg=COLORS["text"],
                       relief=tk.FLAT, font=("Segoe UI", 7))
        e_e.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 2))
        tk.Button(row_e_dir, text="Browse", command=self._pick_engine_dir,
                  bg="#3a3a3a", fg=COLORS["text"], relief=tk.FLAT, padx=4, font=("Segoe UI", 7)).pack(side=tk.RIGHT)

        row_e_file = tk.Frame(parent, bg=COLORS["panel"])
        row_e_file.pack(fill=tk.X, padx=6, pady=(0, 1))
        self._engine_file_var = tk.StringVar(value=getattr(self, "_engine_file", ""))
        self._engine_combo = ttk.Combobox(row_e_file, textvariable=self._engine_file_var,
                                          state="readonly", width=20)
        self._engine_combo.pack(side=tk.RIGHT)
        self._engine_combo.bind("<<ComboboxSelected>>", self._on_engine_change)
        self._refresh_engine_combo()

        # --- Piece style folder ---
        row_p = tk.Frame(parent, bg=COLORS["panel"])
        row_p.pack(fill=tk.X, padx=6, pady=(2, 0))
        tk.Label(row_p, text="Pieces", bg=COLORS["panel"], fg=COLORS["text"],
                 font=("Segoe UI", 7)).pack(side=tk.LEFT)
        self._pieces_dir_var = tk.StringVar(value=getattr(self, "_pieces_dir", str(Path(MDL_DIR).parent.parent / "pieces")))
        e_p = tk.Entry(row_p, textvariable=self._pieces_dir_var, bg="#333", fg=COLORS["text"],
                       relief=tk.FLAT, font=("Segoe UI", 7))
        e_p.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 2))
        tk.Button(row_p, text="Browse", command=self._pick_pieces_dir,
                  bg="#3a3a3a", fg=COLORS["text"], relief=tk.FLAT, padx=4, font=("Segoe UI", 7)).pack(side=tk.RIGHT)

        # --- Torch folder ---
        row_t = tk.Frame(parent, bg=COLORS["panel"])
        row_t.pack(fill=tk.X, padx=6, pady=(2, 0))
        tk.Label(row_t, text="Torch", bg=COLORS["panel"], fg=COLORS["text"],
                 font=("Segoe UI", 7)).pack(side=tk.LEFT)
        self._torch_dir_var = tk.StringVar(value=getattr(self, "_torch_dir", ""))
        e_t = tk.Entry(row_t, textvariable=self._torch_dir_var, bg="#333", fg=COLORS["text"],
                       relief=tk.FLAT, font=("Segoe UI", 7))
        e_t.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 2))
        def _pick_torch_dir():
            d = filedialog.askdirectory(title="Torch package folder")
            if d:
                self._torch_dir = d
                self._torch_dir_var.set(d)
                self._save_config()
        tk.Button(row_t, text="Browse", command=_pick_torch_dir,
                  bg="#3a3a3a", fg=COLORS["text"], relief=tk.FLAT, padx=4, font=("Segoe UI", 7)).pack(side=tk.RIGHT)

        # --- Device selector ---
        row_dev = tk.Frame(parent, bg=COLORS["panel"])
        row_dev.pack(fill=tk.X, padx=6, pady=(2, 0))
        tk.Label(row_dev, text="Device", bg=COLORS["panel"], fg=COLORS["text"],
                 font=("Segoe UI", 7)).pack(side=tk.LEFT)
        self._device_var = tk.StringVar(value=getattr(self, "_device_choice", "auto"))
        dev_combo = ttk.Combobox(row_dev, textvariable=self._device_var,
                                 values=["auto", "cpu", "cuda"], state="readonly", width=8)
        dev_combo.pack(side=tk.RIGHT)
        dev_combo.bind("<<ComboboxSelected>>", self._on_device_change)

        # --- Colors ---
        colors_frame = tk.Frame(parent, bg=COLORS["panel"])
        colors_frame.pack(fill=tk.X, padx=6, pady=(3, 2))
        tk.Label(colors_frame, text="Board colors", bg=COLORS["panel"], fg=COLORS["text"],
                 font=("Segoe UI", 7)).pack(anchor=tk.W)

        c_row = tk.Frame(colors_frame, bg=COLORS["panel"])
        c_row.pack(fill=tk.X)
        for label, attr, default in [
            ("Light", "_light_color", "#E9E5D4"),
            ("Dark", "_dark_color", "#6A994F"),
            ("My arrow", "_arrow_my_color", "#58D90F"),
            ("Opponent arrow", "_arrow_opp_color", "#E53935"),
        ]:
            self._add_color_button(c_row, label, attr, default)

    def _add_color_button(self, parent: tk.Frame, label: str, attr: str, default: str) -> None:
        from chessai import board_widget as bw
        import tkinter.colorchooser as colorchooser

        current = getattr(self, attr, None) or default
        swatch = tk.Frame(parent, bg=current, width=16, height=16, highlightthickness=1,
                          highlightbackground="#555")
        swatch.pack(side=tk.LEFT, padx=(0, 2))
        swatch.pack_propagate(False)

        def pick():
            nonlocal current
            _, hex_color = colorchooser.askcolor(title=label, color=current)
            if not hex_color:
                return
            current = hex_color
            swatch.configure(bg=hex_color)
            setattr(self, attr, hex_color)
            mapping = {
                "_light_color": "BOARD_LIGHT",
                "_dark_color": "BOARD_DARK",
                "_arrow_my_color": "ARROW_MY_COLOR",
                "_arrow_opp_color": "ARROW_OPP_COLOR",
            }
            setattr(bw, mapping[attr], hex_color)
            self._save_config()
            self.board_widget._draw()

        tk.Label(parent, text=label, bg=COLORS["panel"], fg=COLORS["text"],
                 font=("Segoe UI", 7)).pack(side=tk.LEFT, padx=(0, 4))
        swatch.bind("<Button-1>", lambda e: pick())

    def _refresh_model_combo(self) -> None:
        pts = sorted(f.name for f in Path(self._models_dir).glob("*.pt"))
        self._model_combo["values"] = pts
        if pts:
            cur = self._model_file_var.get() or getattr(self, "_model_file", "")
            self._model_file_var.set(cur if cur in pts else pts[0])
            self._apply_model(pts[0] if cur not in pts else cur)

    def _refresh_engine_combo(self) -> None:
        exes = sorted(f.name for f in Path(self._engines_dir).glob("stockfish*"))
        self._engine_combo["values"] = exes
        if exes:
            cur = self._engine_file_var.get() or getattr(self, "_engine_file", "")
            self._engine_file_var.set(cur if cur in exes else exes[0])
            self._apply_engine(exes[0] if cur not in exes else cur)

    def _apply_model(self, name: str) -> None:
        import chessai.classifier as clf
        clf.CURRENT_UNET_MODEL = str(Path(self._models_dir) / name)
        self._model_file = name
        self._save_config()
        self.status_var.set(f"Model: {name}")

    def _apply_engine(self, name: str) -> None:
        self.engine.set_engine_path(str(Path(self._engines_dir) / name))
        self._engine_file = name
        self._save_config()
        self.status_var.set(f"Engine: {name}")

    def _on_model_change(self, event=None) -> None:
        self._apply_model(self._model_file_var.get())

    def _on_engine_change(self, event=None) -> None:
        self._apply_engine(self._engine_file_var.get())

    def _on_device_change(self, event=None) -> None:
        from chessai import classifier as clf
        choice = self._device_var.get()
        self._device_choice = choice
        if choice == "auto":
            clf.FORCE_DEVICE = None
        else:
            clf.FORCE_DEVICE = choice
        self._save_config()
        self.status_var.set(f"Device: {choice}")

    def _on_engine_config_change(self, event=None) -> None:
        self._engine_threads = self._threads_var.get()
        self._engine_hash = self._hash_var.get()
        self.engine.set_threads(self._engine_threads)
        self.engine.set_hash(self._engine_hash)
        self._save_config()
        self.status_var.set(f"Stockfish: {self._engine_threads} threads, {self._engine_hash} MB Hash")

    def _pick_model_dir(self) -> None:
        d = tk.filedialog.askdirectory(title="Models folder (.pt)", initialdir=self._model_dir_var.get())
        if not d:
            return
        self._model_dir_var.set(d)
        self._models_dir = d
        self._model_file = ""
        self._save_config()
        self._refresh_model_combo()

    def _pick_engine_dir(self) -> None:
        d = tk.filedialog.askdirectory(title="Engines folder (stockfish)", initialdir=self._engine_dir_var.get())
        if not d:
            return
        self._engine_dir_var.set(d)
        self._engines_dir = d
        self._engine_file = ""
        self._save_config()
        self._refresh_engine_combo()

    def _pick_pieces_dir(self) -> None:
        d = tk.filedialog.askdirectory(title="Piece style folder (PNG)", initialdir=self._pieces_dir_var.get())
        if not d:
            return
        self._pieces_dir_var.set(d)
        self._pieces_dir = d
        from chessai import board_widget as bw
        bw.CURRENT_PIECE_DIR = d
        self.board_widget._load_piece_images(self.board_widget.square_size)
        self.board_widget._draw()
        self._save_config()

    def _set_always_on_top(self, enabled: bool) -> None:
        self.pinned_var.set(enabled)
        self.root.attributes("-topmost", enabled)
        if enabled:
            self.pin_btn.config(
                text="📌 Always on top",
                bg=COLORS["accent"],
                fg="#ffffff",
                activebackground=COLORS["accent"],
            )
            self.root.lift()
        else:
            self.pin_btn.config(
                text="📌 Pin",
                bg="#3a3a3a",
                fg=COLORS["text"],
                activebackground="#4a4a4a",
            )

    def _toggle_pin(self) -> None:
        self._set_always_on_top(not self.pinned_var.get())

    def _set_player_color(self, color: chess.Color) -> None:
        self.player_color = color
        self._sync_board_orientation()
        self.screen_reader.set_orientation_for_player(color)
        self._refresh_player_buttons()
        self._request_analysis()

    def _on_turn_mode_change(self) -> None:
        self._sync_turn_mode()
        self._refresh_turn_buttons()
        if not self.auto_turn_var.get():
            self._apply_turn_to_board(self._current_manual_turn())
        self._request_analysis()

    def _current_manual_turn(self) -> chess.Color:
        return chess.BLACK if self.turn_to_move_var.get() == "black" else chess.WHITE

    def _set_turn_to_move(self, color: chess.Color) -> None:
        self.turn_to_move_var.set("black" if color == chess.BLACK else "white")
        self.auto_turn_var.set(False)
        self._sync_turn_mode()
        self._refresh_turn_buttons()
        self._apply_turn_to_board(color)
        self._request_analysis()

    def _sync_turn_mode(self) -> None:
        self.screen_reader.set_turn_mode(
            auto_turn=self.auto_turn_var.get(),
            manual_turn=self._current_manual_turn(),
        )

    def _apply_turn_to_board(self, turn: chess.Color) -> None:
        parts = self.board.fen().split()
        if len(parts) < 2:
            return
        parts[1] = "w" if turn == chess.WHITE else "b"
        try:
            self.board = chess.Board(" ".join(parts[:6]))
        except ValueError:
            return
        self.screen_reader.apply_manual_turn()
        self.board_widget.set_board(self.board, keep_arrows=True)
        self.fen_var.set(self.board.fen())
        turn_name = "White" if turn == chess.WHITE else "Black"
        self.turn_status_var.set(f"Manual: {turn_name} to move")

    def _refresh_turn_buttons(self) -> None:
        auto = self.auto_turn_var.get()
        active = {"bg": COLORS["accent"], "fg": "#ffffff", "activebackground": COLORS["accent"]}
        idle = {"bg": "#3a3a3a", "fg": COLORS["text"], "activebackground": "#4a4a4a"}
        disabled = {"bg": "#2a2a2a", "fg": "#666666", "activebackground": "#2a2a2a"}

        if auto:
            self.turn_white_btn.config(**disabled, state=tk.DISABLED)
            self.turn_black_btn.config(**disabled, state=tk.DISABLED)
            turn_name = "White" if self.board.turn == chess.WHITE else "Black"
            self.turn_status_var.set(f"Auto: {turn_name} to move")
        else:
            self.turn_white_btn.config(state=tk.NORMAL)
            self.turn_black_btn.config(state=tk.NORMAL)
            if self._current_manual_turn() == chess.WHITE:
                self.turn_white_btn.config(**active)
                self.turn_black_btn.config(**idle)
            else:
                self.turn_white_btn.config(**idle)
                self.turn_black_btn.config(**active)
            turn_name = "White" if self._current_manual_turn() == chess.WHITE else "Black"
            self.turn_status_var.set(f"Manual: {turn_name} to move")

    def _sync_board_orientation(self) -> None:
        """Your pieces should always appear at the bottom of the preview board."""
        self.board_widget.set_flip(self.player_color == chess.BLACK)

    def _refresh_player_buttons(self) -> None:
        active = {"bg": COLORS["accent"], "fg": "#ffffff", "activebackground": COLORS["accent"]}
        idle = {"bg": "#3a3a3a", "fg": COLORS["text"], "activebackground": "#4a4a4a"}
        orient = "board as on screen (black at bottom)" if self.player_color == chess.BLACK else "board as on screen (white at bottom)"
        if self.player_color == chess.WHITE:
            self.white_btn.config(**active)
            self.black_btn.config(**idle)
            self.player_status_var.set(f"You play White — {orient}")
        else:
            self.white_btn.config(**idle)
            self.black_btn.config(**active)
            self.player_status_var.set(f"You play Black — {orient}")

    def _toggle_arrows(self) -> None:
        self.board_widget.set_show_arrows(self.show_arrows_var.get())
        if self.show_arrows_var.get():
            self._request_analysis()

    def _build_suggestion_arrows(self, board: chess.Board, pv: list[chess.Move]) -> list[MoveArrow]:
        if len(pv) < 1:
            return []

        arrows: list[MoveArrow] = []
        my_turn = board.turn == self.player_color

        if my_turn:
            arrows.append(MoveArrow(pv[0], MY_ARROW, "your move"))
            if len(pv) > 1:
                arrows.append(MoveArrow(pv[1], OPP_ARROW, "opponent's reply"))
        else:
            arrows.append(MoveArrow(pv[0], OPP_ARROW, "opponent's move"))
            if len(pv) > 1:
                arrows.append(MoveArrow(pv[1], MY_ARROW, "your reply"))

        return arrows

    def _start_engine(self) -> None:
        if not self.engine.is_running:
            try:
                if Path(self.engine.engine_path).is_file():
                    self.engine.start()
                    self.engine.set_threads(self._threads_var.get())
                    self.engine.set_hash(self._hash_var.get())
                    backend = match_backend_name()
                    backend_label = {"cuda": "GPU CUDA", "opencl": "GPU OpenCL", "cpu": "CPU"}.get(
                        backend, backend
                    )
                    self.status_var.set(f"Stockfish ready | vision: {backend_label}")
                else:
                    self.status_var.set("Engine not selected — specify folder in settings")
            except FileNotFoundError as exc:
                self.status_var.set(str(exc))

    def _on_board_move(self, move: chess.Move) -> None:
        if self.screen_watch:
            return
        self._undo_stack.append(self.board.fen())
        self.board = self.board_widget.board
        self.fen_var.set(self.board.fen())
        self._request_analysis()

    def _save_to_dataset(self, reading: ScreenReadResult) -> None:
        """Save scan result to dataset/ for future AI training."""
        import json
        import time
        from pathlib import Path

        dataset_dir = Path(__file__).resolve().parent.parent / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        ts = int(time.time())
        board_fen = reading.board_fen
        img_path = dataset_dir / f"board_{ts}.png"
        frame = self.screen_reader.capture()
        if reading.flipped:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        cv2.imwrite(str(img_path), frame)

        # Build cell labels from FEN
        files = "abcdefgh"
        cells = {}
        for rank_idx, row in enumerate(board_fen.split("/")):
            rank = 8 - rank_idx
            file_idx = 0
            for ch in row:
                if ch.isdigit():
                    file_idx += int(ch)
                else:
                    cells[f"{files[file_idx]}{rank}"] = ch
                    file_idx += 1

        sample = {"image": img_path.name, "fen": board_fen, "cells": cells}
        meta_path = dataset_dir / "metadata.json"
        samples = []
        if meta_path.exists():
            with open(meta_path) as f:
                samples = json.load(f)
        samples.append(sample)
        with open(meta_path, "w") as f:
            json.dump(samples, f, indent=1)

    def _apply_screen_reading(self, reading: ScreenReadResult) -> bool:
        fen = f"{reading.board_fen} {'w' if reading.turn == chess.WHITE else 'b'} - - 0 1"
        try:
            new_board = chess.Board(fen)
        except ValueError as e:
            self.status_var.set(f"Scan: invalid FEN {e}")
            return False
        if not new_board.is_valid():
            # Try the other turn — infer_turn_from_position can be wrong on checks
            try:
                other_turn = not reading.turn
                alt_fen = f"{reading.board_fen} {'w' if other_turn == chess.WHITE else 'b'} - - 0 1"
                new_board = chess.Board(alt_fen)
                if not new_board.is_valid():
                    self.status_var.set("Scan: illegal position")
                    return False
                reading.turn = other_turn
            except ValueError:
                self.status_var.set("Scan: illegal position")
                return False
        if (
            new_board.board_fen() == self.board.board_fen()
            and new_board.turn == self.board.turn
            and not reading.changed
        ):
            return False

        # Correct turn and populate move info via move detection
        if reading.changed:
            prev_fen = self.board.board_fen()
            prev_turn = self.board.turn
            if prev_fen != reading.board_fen:
                from chessai.position_tracker import detect_move

                move_info = detect_move(prev_fen, prev_turn, reading.board_fen)
                if move_info is not None:
                    reading.turn = move_info.turn
                    if not reading.last_move_san:
                        reading.last_move_san = move_info.san

        self.board = new_board
        self._sync_board_orientation()
        self.board_widget.set_board(self.board, keep_arrows=True)
        self.board_widget.set_last_move(reading.last_move)
        self.fen_var.set(self.board.fen())

        if self.collect_var.get() and reading.changed:
            self._save_to_dataset(reading)

        if self.auto_turn_var.get():
            self.turn_to_move_var.set("black" if reading.turn == chess.BLACK else "white")
            self._refresh_turn_buttons()

        turn_name = "White" if reading.turn == chess.WHITE else "Black"
        move_note = ""
        if reading.last_move_san:
            mover = "Black" if reading.turn == chess.WHITE else "White"
            move_note = f" | {mover} moved: {reading.last_move_san}"
        self.screen_status_var.set(
            f"Vision ({reading.source}): {turn_name} to move (move {reading.ply}){move_note} | {reading.confidence:.0%}"
        )

        if self.auto_play_var.get() and self.screen_watch:
            if reading.turn == self.player_color and reading.changed:
                self._pending_auto_move = True
                self.status_var.set("Auto-play: analyzing for best move…")
            elif reading.turn != self.player_color:
                self._pending_auto_move = False

        self._analyze_async()

        # Track position history for threefold repetition avoidance
        pos_key = " ".join(self.board.fen().split()[:4])
        self._fen_history.append(pos_key)
        max_size = self._history_size_var.get()
        if max_size < 10:
            max_size = 100
        while len(self._fen_history) > max_size:
            self._fen_history.pop(0)

        return True

    def _new_game(self) -> None:
        self._fen_history.clear()
        self.board = chess.Board()
        self._undo_stack.clear()
        self.board_widget.set_board(self.board)
        self.board_widget.set_last_move(None)
        self.board_widget.set_suggestion_arrows([])
        self.fen_var.set(self.board.fen())
        self._request_analysis()

    def _undo(self) -> None:
        if not self._undo_stack:
            return
        fen = self._undo_stack.pop()
        self.board = chess.Board(fen)
        self.board_widget.set_board(self.board)
        self.fen_var.set(self.board.fen())
        self._request_analysis()

    def _flip_board(self) -> None:
        self.board_widget.set_flip(not self.board_widget.flip)
        self.screen_reader.screen_flipped = self.board_widget.flip
        self.screen_reader.tracker.reset()
        self._request_analysis()

    def _apply_fen(self) -> None:
        fen = self.fen_var.get().strip()
        try:
            self.board = chess.Board(fen)
        except ValueError as exc:
            messagebox.showerror("FEN", f"Invalid FEN: {exc}")
            return
        self.board_widget.set_board(self.board)
        self.board_widget.set_last_move(None)
        self._request_analysis()

    def _load_fen_dialog(self) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.title("Load FEN")
        dialog.configure(bg=COLORS["panel"])
        dialog.geometry("520x120")
        text = tk.Text(dialog, height=3, bg="#333333", fg=COLORS["text"], relief=tk.FLAT)
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text.insert("1.0", self.fen_var.get())

        def apply() -> None:
            self.fen_var.set(text.get("1.0", tk.END).strip())
            dialog.destroy()
            self._apply_fen()

        tk.Button(dialog, text="OK", command=apply, bg=COLORS["accent"], fg="#fff", relief=tk.FLAT).pack(
            pady=6
        )

    def _analyze_once(self) -> None:
        self.continuous = False
        self.engine.stop_continuous_analysis()
        self._analyze_async()

    def _toggle_continuous(self) -> None:
        self.continuous = not self.continuous
        if self.continuous:
            self._restart_continuous()
            self.status_var.set("Continuous analysis on")
        else:
            self.engine.stop_continuous_analysis()
            self.status_var.set("Continuous analysis off")

    def _restart_continuous(self) -> None:
        self.engine.stop_continuous_analysis()

        def on_update(result: AnalysisResult) -> None:
            self.analysis_queue.put((self._analysis_token, result))

        dv = self.depth_var.get()
        self.engine.start_continuous_analysis(
            get_board=lambda: self.board,
            on_update=on_update,
            time_limit=float(self.time_var.get()),
            depth=dv if dv > 0 else None,
            multipv=int(self.multipv_var.get()),
            poll_interval=0.2,
        )

    def _request_analysis(self) -> None:
        if self.continuous:
            return
        self._analyze_async()

    def _analyze_async(self) -> None:
        board_copy = self.board.copy()
        time_limit = float(self.time_var.get())
        dv = self.depth_var.get()
        depth = dv if dv > 0 else None
        try:
            multipv = int(self.multipv_var.get())
        except (ValueError, tk.TclError):
            multipv = 3
        self._analysis_token += 1
        token = self._analysis_token
        self._pending_analysis = True

        def worker() -> None:
            try:
                result = self.engine.analyze_position(
                    board_copy, time_limit=time_limit, depth=depth, multipv=multipv
                )
                result.finished = True
                self.analysis_queue.put((token, result))
            except Exception as exc:
                self.analysis_queue.put(
                    (token, AnalysisResult(fen=board_copy.fen(), lines=[], finished=True))
                )
                self.root.after(0, lambda e=exc: self.status_var.set(f"Error: {e}"))

        threading.Thread(target=worker, daemon=True).start()
        self.status_var.set("Analyzing…")

    def _poll_analysis(self) -> None:
        try:
            while True:
                token, result = self.analysis_queue.get_nowait()
                self._show_analysis(token, result)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_analysis)

    def _show_analysis(self, token: int, result: AnalysisResult) -> None:
        if not self.continuous and token != self._analysis_token:
            return

        result_board = chess.Board(result.fen)
        if result_board.board_fen() != self.board.board_fen():
            return

        self._pending_analysis = False

        if not result.lines:
            self.status_var.set("No analysis data")
            return

        top = result.lines[0]
        board = self.board
        self._update_eval_bar(top, board.turn)
        arrows = (
            self._build_suggestion_arrows(board, top.pv)
            if self.show_arrows_var.get() and top.pv
            else []
        )
        self.board_widget.set_suggestion_arrows(arrows)

        for child in self.lines_frame.winfo_children():
            child.destroy()

        for line in result.lines:
            frame = tk.Frame(self.lines_frame, bg="#333333", pady=3, padx=6)
            frame.pack(fill=tk.X, pady=2)
            tag = ""
            if line.rank == 1:
                if board.turn == self.player_color:
                    tag = "  ← your move"
                else:
                    tag = "  ← opponent's move"
            header = f"#{line.rank}  {line.score_text}  (d{line.depth}){tag}"
            tk.Label(
                frame,
                text=header,
                font=("Consolas", 10, "bold"),
                bg="#333333",
                fg=COLORS["accent"],
                anchor=tk.W,
            ).pack(fill=tk.X)
            pv = line.pv_san(board)
            tk.Label(
                frame,
                text=pv or "—",
                font=("Segoe UI", 9),
                bg="#333333",
                fg=COLORS["text"],
                wraplength=300,
                justify=tk.LEFT,
                anchor=tk.W,
            ).pack(fill=tk.X)

        self.status_var.set(
            f"Depth {top.depth} | nodes {top.nodes:,}"
        )

        if self._pending_auto_move and self.auto_play_var.get() and result.lines:
            self._pending_auto_move = False
            best_move = None
            move_san = ""
            for line in result.lines:
                if not line.pv:
                    continue
                candidate = line.pv[0]
                test_board = board.copy()
                test_board.push(candidate)
                pos_key = " ".join(test_board.fen().split()[:4])
                if self._fen_history.count(pos_key) >= 2:
                    continue
                best_move = candidate
                try:
                    move_san = board.san(best_move)
                except ValueError:
                    move_san = str(best_move)
                break
            if best_move is None:
                best_move = top.pv[0]
                try:
                    move_san = board.san(best_move)
                except ValueError:
                    move_san = str(best_move)
            self.status_var.set(f"Auto-play: {move_san}…")
            self._execute_auto_move(best_move, move_san)

    def _update_eval_bar(self, line, turn: chess.Color) -> None:
        self.eval_canvas.delete("all")
        h = 160
        if line.score_mate is not None:
            mate = line.score_mate
            if turn != self.player_color:
                mate = -mate
            text = f"M{mate}"
            fill_ratio = 1.0 if mate > 0 else 0.0
        elif line.score_cp is not None:
            cp = line.score_cp if turn == chess.WHITE else -line.score_cp
            if self.player_color == chess.BLACK:
                cp = -cp
            text = f"{cp / 100:+.2f}"
            fill_ratio = 0.5 + max(-1.0, min(1.0, cp / 800)) * 0.5
        else:
            text = "0.00"
            fill_ratio = 0.5

        white_h = int(h * fill_ratio)
        self.eval_canvas.create_rectangle(0, 0, 24, h - white_h, fill=COLORS["eval_black"], outline="")
        self.eval_canvas.create_rectangle(
            0, h - white_h, 24, h, fill=COLORS["eval_white"], outline=""
        )
        self.eval_label.config(text=text)

    def _execute_auto_move(self, move: chess.Move, move_san: str) -> None:
        if not self.screen_reader.is_calibrated or not self.screen_reader.region:
            return
        delay = float(self.move_delay_var.get())
        threading.Thread(
            target=self._auto_move_worker,
            args=(move, delay, move_san),
            daemon=True,
        ).start()

    def _auto_move_worker(self, move: chess.Move, delay: float, move_san: str) -> None:
        import time

        from chessai import win32_automation as win32

        if delay > 0:
            time.sleep(delay)

        hwnd = self._auto_play_hwnd
        if hwnd is None:
            target = self.target_window_var.get().strip()
            if target:
                hwnd = win32.find_window(target)
                if hwnd:
                    self._auto_play_hwnd = hwnd

        if hwnd is None:
            self.root.after(0, lambda: self.status_var.set("Auto-play: target window not found"))
            return

        try:
            win32.make_move(
                move,
                self.screen_reader.region,
                self.screen_reader.screen_flipped,
                hwnd=hwnd,
                click_delay=0.02,
            )
            self.root.after(0, lambda: self.status_var.set(f"Auto-play: {move_san} done"))
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"Auto-play error: {e}"))

    def _detect_target_window(self) -> None:
        from chessai import win32_automation as win32

        title = win32.get_foreground_window_title()
        if title:
            self.target_window_var.set(title)
            self._auto_play_hwnd = win32.get_foreground_window()
            self.status_var.set(f"Target window: {title}")

    def _on_auto_play_toggle(self) -> None:
        if self.auto_play_var.get():
            self._detect_target_window()
            self.status_var.set("Auto-play enabled")
        else:
            self._pending_auto_move = False
            self.status_var.set("Auto-play disabled")

    def _scan_now(self) -> None:
        if not self.screen_reader.is_calibrated:
            self._select_screen_region(then_watch=False)
            return
        reading = self.screen_reader.scan_once(self.player_color)
        if reading and self._apply_screen_reading(reading):
            self.status_var.set("Scan complete")
        else:
            messagebox.showwarning(
                "Scan",
                "Could not recognize the position.\n"
                "Select the region more precisely (only the board cells) and try again.",
            )

    def _reset_region(self) -> None:
        self.screen_reader.reset_region()
        self.status_var.set("Click and drag to select the board region...")
        self.root.update()
        self._select_screen_region(then_watch=False)

    def _toggle_auto_screen(self) -> None:
        if self.screen_watch:
            self._stop_screen_watch()
            return
        if not self.screen_reader.is_calibrated:
            self._select_screen_region(then_watch=True)
            return
        self._start_screen_watch()

    def _start_screen_watch(self) -> None:
        if not self.screen_reader.is_calibrated:
            messagebox.showwarning(
                "Auto from screen",
                "First select the board on chess.com:\n"
                "press '▶ Auto from screen' with the starting position open.",
            )
            return

        self.screen_watch = True
        self.board_widget.set_interactive(False)
        self.auto_mode_var.set(True)
        self.auto_screen_btn.config(
            text="■ Stop auto",
            bg="#c62828",
            activebackground="#c62828",
        )

        if self.continuous:
            self.continuous = False
            self.engine.stop_continuous_analysis()

        self.screen_status_var.set("Vision: waiting for board change…")
        self.status_var.set("Auto from screen — fast local scan")
        self._screen_tick(force=True)

    def _on_async_screen_read(self, reading: Optional[ScreenReadResult], force: bool, error: Optional[str] = None) -> None:
        if not self.screen_watch:
            return
        if error:
            if error != "busy":
                self.status_var.set(f"Scan: {error}")
            self._screen_job = self.root.after(50, lambda: self._screen_tick(False))
            return
        if reading:
            self._apply_screen_reading(reading)
        self._screen_job = self.root.after(50, lambda: self._screen_tick(False))

    def _screen_tick(self, force: bool = False) -> None:
        if not self.screen_watch:
            return

        # Watchdog: reset stuck scan after 5s
        if (
            self.screen_reader._scanning
            and self.screen_reader._scan_start_time > 0
            and time.monotonic() - self.screen_reader._scan_start_time > 5.0
        ):
            with self.screen_reader._scan_lock:
                self.screen_reader._scanning = False
                self.screen_reader._scan_start_time = 0.0
            self.status_var.set("Scan: reset stuck scan")

        def deliver(result: Optional[ScreenReadResult], err: Optional[str] = None) -> None:
            self.root.after(0, lambda: self._on_async_screen_read(result, force, err))

        try:
            self.screen_reader.read_position_async(
                self.player_color,
                deliver,
                force=force,
            )
        except Exception as exc:
            self.status_var.set(f"Scan error: {exc}")
            self._screen_job = self.root.after(500, lambda: self._screen_tick(False))

    def _stop_screen_watch(self) -> None:
        self.screen_watch = False
        self.board_widget.set_interactive(True)
        self.auto_mode_var.set("")
        self.auto_screen_btn.config(
            text="▶ Auto from screen",
            bg=COLORS["accent"],
            activebackground=COLORS["accent"],
        )
        if self._screen_job:
            self.root.after_cancel(self._screen_job)
            self._screen_job = None
        self.screen_status_var.set("Screen: inactive")
        self.status_var.set("Auto from screen off")

    def _select_screen_region(self, *, then_watch: bool = False) -> None:
        def on_done(region) -> None:
            if region is None:
                return
            self.screen_reader.set_region(region)
            self.screen_reader.screen_flipped = self.player_color == chess.BLACK
            try:
                reading = self.screen_reader.scan_once(self.player_color)
            except Exception as exc:
                messagebox.showerror("Scan", f"Scanning error:\n{exc}")
                return
            if reading:
                self._apply_screen_reading(reading)
                if then_watch:
                    self._start_screen_watch()
                else:
                    messagebox.showinfo(
                        "Scan",
                        "Board region saved.\n"
                        "Press '▶ Auto from screen' to track moves.",
                    )
            else:
                messagebox.showwarning(
                    "Scan",
                    "Region saved, but position not recognized.\n"
                    "Select only the board cells and make sure it's fully visible.",
                )

        self.root.after(200, lambda: select_screen_region(self.root, on_done))

    def _toggle_screen_watch(self) -> None:
        self._toggle_auto_screen()

    def _toggle_clipboard_watch(self) -> None:
        self.clipboard_watch = not self.clipboard_watch
        if self.clipboard_watch:
            self._last_clipboard = self.root.clipboard_get() if self._try_clipboard() else ""
            self.status_var.set("FEN clipboard watch enabled")
            self._clipboard_tick()
        else:
            if self._clipboard_job:
                self.root.after_cancel(self._clipboard_job)
                self._clipboard_job = None
            self.status_var.set("Clipboard watch disabled")

    def _try_clipboard(self) -> bool:
        try:
            self.root.clipboard_get()
            return True
        except tk.TclError:
            return False

    def _clipboard_tick(self) -> None:
        if not self.clipboard_watch:
            return
        try:
            text = self.root.clipboard_get().strip()
        except tk.TclError:
            text = ""
        if text and text != self._last_clipboard and FEN_RE.match(text):
            self._last_clipboard = text
            self.fen_var.set(text)
            self._apply_fen()
        self._clipboard_job = self.root.after(400, self._clipboard_tick)

    def _on_close(self) -> None:
        self._stop_screen_watch()
        self.clipboard_watch = False
        self.screen_reader.close()
        self.engine.stop()
        uninstall_hotkey()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    ChessAIApp().run()


if __name__ == "__main__":
    main()
