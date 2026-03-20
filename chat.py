"""
Chat window for local Qwen2.5-0.5B running on GPU via WGPU.

Launches claude-rlm in interactive mode (model stays loaded between messages).
Tokens stream to the window in real-time as they're generated.
"""

import subprocess
import threading
import tkinter as tk
from tkinter import scrolledtext
import time
import sys
import os

BINARY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "target", "release", "claude-rlm.exe")
if not os.path.exists(BINARY):
    BINARY = "claude-rlm"


class ChatApp:
    def __init__(self, root):
        self.root = root
        root.title("Local Brain  |  Qwen 0.5B @ WGPU  |  no cloud, no CUDA")
        root.configure(bg="#1e1e2e")
        root.geometry("750x560")

        # Chat display
        self.chat = scrolledtext.ScrolledText(
            root, wrap=tk.WORD, font=("Consolas", 11),
            bg="#1e1e2e", fg="#cdd6f4", insertbackground="#cdd6f4",
            selectbackground="#45475a", relief=tk.FLAT, padx=12, pady=12,
            state=tk.DISABLED,
        )
        self.chat.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 0))

        self.chat.tag_config("user", foreground="#89b4fa", font=("Consolas", 11, "bold"))
        self.chat.tag_config("bot", foreground="#a6e3a1")
        self.chat.tag_config("system", foreground="#6c7086", font=("Consolas", 9, "italic"))
        self.chat.tag_config("speed", foreground="#f9e2af", font=("Consolas", 9))

        # Input area
        input_frame = tk.Frame(root, bg="#313244")
        input_frame.pack(fill=tk.X, padx=8, pady=8)

        self.entry = tk.Entry(
            input_frame, font=("Consolas", 12),
            bg="#313244", fg="#cdd6f4", insertbackground="#cdd6f4",
            relief=tk.FLAT,
        )
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 4), pady=8)
        self.entry.bind("<Return>", self.on_send)
        self.entry.focus()

        send_btn = tk.Button(
            input_frame, text="Send", command=self.on_send,
            font=("Consolas", 10, "bold"),
            bg="#89b4fa", fg="#1e1e2e", activebackground="#74c7ec",
            relief=tk.FLAT, padx=16, pady=4,
        )
        send_btn.pack(side=tk.RIGHT, padx=(4, 8), pady=8)

        self.generating = False
        self.proc = None
        self.append_system("Starting engine...")

        # Launch the persistent backend
        threading.Thread(target=self.start_engine, daemon=True).start()

    def append_system(self, text):
        self.append_msg("system", text)

    def append_msg(self, tag, text):
        self.chat.config(state=tk.NORMAL)
        self.chat.insert(tk.END, text + "\n", tag)
        self.chat.see(tk.END)
        self.chat.config(state=tk.DISABLED)

    def append_streaming(self, text, tag="bot"):
        """Append text without a trailing newline (for streaming tokens)."""
        self.chat.config(state=tk.NORMAL)
        self.chat.insert(tk.END, text, tag)
        self.chat.see(tk.END)
        self.chat.config(state=tk.DISABLED)

    def start_engine(self):
        """Launch claude-rlm chat --interactive as a persistent subprocess."""
        try:
            self.proc = subprocess.Popen(
                [BINARY, "chat", "--interactive"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,  # binary mode for clean EOT handling
            )

            # Read stderr until we see READY
            while True:
                line = self.proc.stderr.readline()
                if not line:
                    self.root.after(0, self.append_system, "Engine died during startup")
                    return
                line_str = line.decode("utf-8", errors="replace").strip()
                if "READY" in line_str:
                    break
                # Show loading progress
                self.root.after(0, self.append_system, line_str)

            self.root.after(0, self.append_system,
                "Ready! Model loaded on GPU. Type a message.\n")

        except FileNotFoundError:
            self.root.after(0, self.append_system,
                f"Could not find {BINARY}. Run: cargo build --release")
        except Exception as e:
            self.root.after(0, self.append_system, f"Engine error: {e}")

    def on_send(self, event=None):
        if self.generating or not self.proc or self.proc.poll() is not None:
            return
        msg = self.entry.get().strip()
        if not msg:
            return
        self.entry.delete(0, tk.END)
        self.append_msg("user", f"You: {msg}")
        self.generating = True
        threading.Thread(target=self.generate, args=(msg,), daemon=True).start()

    def generate(self, user_msg):
        t0 = time.perf_counter()
        token_count = 0

        try:
            # Send message to the persistent process
            self.proc.stdin.write((user_msg + "\n").encode("utf-8"))
            self.proc.stdin.flush()

            # Prefix
            self.root.after(0, self.append_streaming, "Brain: ", "bot")

            # Stream tokens until EOT (\x04)
            buf = b""
            while True:
                chunk = self.proc.stdout.read(1)
                if not chunk:
                    break  # process died
                if chunk == b"\x04":
                    # Flush remaining buffer
                    if buf:
                        text = buf.decode("utf-8", errors="replace")
                        token_count += text.count(" ") + 1
                        self.root.after(0, self.append_streaming, text, "bot")
                    break

                buf += chunk

                # Flush on spaces/newlines for smooth streaming
                if chunk in (b" ", b"\n", b".", b",", b"!", b"?"):
                    text = buf.decode("utf-8", errors="replace")
                    token_count += 1
                    self.root.after(0, self.append_streaming, text, "bot")
                    buf = b""

            elapsed = time.perf_counter() - t0
            self.root.after(0, self.append_msg, "speed", f"\n  [{elapsed:.1f}s]")

        except Exception as e:
            self.root.after(0, self.append_system, f"(error: {e})")
        finally:
            self.generating = False

    def on_close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = ChatApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)

    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    root.mainloop()


if __name__ == "__main__":
    main()
