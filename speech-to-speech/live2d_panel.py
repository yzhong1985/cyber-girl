"""
live2d_panel.py

给管线加一个 Live2D 网页形象：浏览器开 http://127.0.0.1:8901/ 显示立绘，
管线播放 TTS 时把音量包络通过 SSE 实时推给页面，驱动嘴巴开合。

· set_mouth(v)  —— 供音频端调用，v ∈ [0,1]，表示当前张嘴程度
· serve_live2d(directory, port=8901) —— 启动服务（守护线程）
"""

import json
import logging
import mimetypes
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

logger = logging.getLogger(__name__)

_state = {"mouth": 0.0}
_lock = threading.Lock()


def set_mouth(v: float):
    with _lock:
        _state["mouth"] = max(0.0, min(1.0, float(v)))


def _get_mouth() -> float:
    with _lock:
        return _state["mouth"]


mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/json", ".json")
mimetypes.add_type("application/octet-stream", ".moc3")


def serve_live2d(directory: str, port: int = 8901):
    root = Path(directory).resolve()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send_file(self, rel):
            fp = (root / rel).resolve()
            if not str(fp).startswith(str(root)) or not fp.is_file():
                self.send_response(404)
                self.end_headers()
                return
            ctype = mimetypes.guess_type(str(fp))[0] or "application/octet-stream"
            data = fp.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = self.path.split("?", 1)[0]

            if path == "/mouth-stream":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                try:
                    while True:
                        payload = json.dumps({"mouth": round(_get_mouth(), 3)})
                        self.wfile.write(f"data: {payload}\n\n".encode())
                        self.wfile.flush()
                        time.sleep(0.04)   # ~25fps
                except (BrokenPipeError, ConnectionResetError):
                    return
                return

            if path == "/" or path == "":
                self._send_file("index.html")
                return

            self._send_file(path.lstrip("/"))

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    logger.info(f"Live2D 形象: http://127.0.0.1:{port}/")
    return srv
