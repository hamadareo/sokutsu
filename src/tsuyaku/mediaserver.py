"""録音を画面に渡す小さな HTTP サーバー(W2, 11/07)。

選んだ理由: 画面は pywebview の内蔵サーバー(http://127.0.0.1:<port>/index.html)から開かれていて、
logs/ はその外にある。<audio> は再生位置を変えるたびに Range 要求で必要な所だけ取りに来るので、
Range に応える HTTP で渡すのがいちばん素直(1時間 15MB を JS に base64 で流すと数秒固まり、
録音中で伸び続ける CAF も渡せない)。127.0.0.1 だけで待ち受け、起動ごとの合言葉(token)が
URL に無い要求と、記録のフォルダの録音(session-*.audio.m4a/.caf)以外は断る。
"""
from __future__ import annotations

import re
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

_NAME = re.compile(r"^session-[0-9A-Za-z_-]+\.audio\.(m4a|caf)$")
_TYPES = {"m4a": "audio/mp4", "caf": "audio/x-caf"}


class MediaServer:
    def __init__(self, root: "str | Path", host: str = "127.0.0.1"):
        self.root = Path(root).resolve()
        self.token = secrets.token_urlsafe(16)
        srv = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):   # noqa: D401 - 静かに
                pass

            def do_HEAD(self):
                self._serve(head=True)

            def do_GET(self):
                self._serve(head=False)

            def _serve(self, head: bool):
                parts = self.path.split("?", 1)[0].split("/")
                # /a/<token>/<name>
                if len(parts) != 4 or parts[1] != "a":
                    return self.send_error(404)
                # 要求の行は latin-1 で来る。str のままだと非 ASCII で compare_digest が例外を出す
                if not secrets.compare_digest(parts[2].encode("utf-8", "surrogateescape"),
                                              srv.token.encode()):
                    return self.send_error(403)
                name = unquote(parts[3])
                p = srv.root / name
                # 記録のフォルダの外を指す別名(シンボリックリンク)は渡さない
                if not _NAME.match(name) or not p.is_file() or p.resolve().parent != srv.root:
                    return self.send_error(404)
                size = p.stat().st_size
                lo, hi = 0, size - 1
                rng = self.headers.get("Range")
                status = 200
                if rng:
                    m = re.match(r"bytes=(\d{0,18})-(\d{0,18})$", rng.strip())
                    if not m or (not m.group(1) and not m.group(2)):
                        return self.send_error(416)
                    if m.group(1):
                        lo = int(m.group(1))
                        hi = int(m.group(2)) if m.group(2) else size - 1
                    else:                         # bytes=-N(最後の N バイト)
                        lo = max(0, size - int(m.group(2)))
                    hi = min(hi, size - 1)
                    if lo >= size or lo > hi:
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{size}")
                        self.end_headers()
                        return
                    status = 206
                self.send_response(status)
                self.send_header("Content-Type", _TYPES[name.rsplit(".", 1)[1]])
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(hi - lo + 1))
                self.send_header("Cache-Control", "no-store")   # 録音中の CAF は伸びる
                if status == 206:
                    self.send_header("Content-Range", f"bytes {lo}-{hi}/{size}")
                self.end_headers()
                if head:
                    return
                try:
                    with p.open("rb") as f:
                        f.seek(lo)
                        left = hi - lo + 1
                        while left > 0:
                            b = f.read(min(65536, left))
                            if not b:
                                break
                            self.wfile.write(b)
                            left -= len(b)
                except (BrokenPipeError, ConnectionResetError):
                    pass   # 再生位置を変えると <audio> は前の要求を切る

        self._httpd = ThreadingHTTPServer((host, 0), H)
        self._httpd.daemon_threads = True
        self.port = self._httpd.server_address[1]
        self.host = host
        self._th = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._th.start()

    def url(self, path: "str | Path") -> str:
        return f"http://{self.host}:{self.port}/a/{self.token}/{Path(path).name}"

    def close(self) -> None:
        try:
            self._httpd.shutdown()
            self._httpd.server_close()
        except Exception:  # noqa: BLE001
            pass
