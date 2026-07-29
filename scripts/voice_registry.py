"""
voice_registry.py

热切换：不重启管线的前提下换音色 / 换人格 / 换 LLM。

设计取舍（重要）：
  · 换音色  —— 零成本。Qwen3-TTS 常驻显存，只换参考音频路径，切换 <10ms。
  · 换人格  —— 零成本。改 system prompt + 清空历史即可。
  · 换 LLM  —— 交给 llama-swap 代理，本进程只改请求里的 model 字段。
                千万别在本进程里 load/unload GGUF，15G 显存扛不住抖动。

启动后访问 http://127.0.0.1:8900/ 就是个切换面板。
"""

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

logger = logging.getLogger(__name__)


class VoiceRegistry:
    def __init__(self, config_path="characters.json"):
        self.lock = threading.Lock()
        self.config_path = Path(config_path)
        self.characters = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.active = next(iter(self.characters))
        self._on_switch = []          # 回调：清空对话历史等

    # ---------- 给 handler 调用 ----------

    def current_ref_audio(self):
        with self.lock:
            return self.characters[self.active]["ref_audio"]

    def current_prompt(self):
        with self.lock:
            return self.characters[self.active]["system_prompt"]

    def current_model(self):
        """给 llama-swap 用的 model 名"""
        with self.lock:
            return self.characters[self.active].get("llm_model", "default")

    # ---------- 切换 ----------

    def register_callback(self, fn):
        self._on_switch.append(fn)

    def switch(self, name: str) -> bool:
        if name not in self.characters:
            return False
        with self.lock:
            self.active = name
        logger.info(f"切换到角色: {name}")
        for fn in self._on_switch:
            try:
                fn(name)
            except Exception as e:
                logger.warning(f"切换回调失败: {e}")
        return True

    def reload_config(self):
        """改完 characters.json 不用重启"""
        with self.lock:
            self.characters = json.loads(self.config_path.read_text(encoding="utf-8"))
            if self.active not in self.characters:
                self.active = next(iter(self.characters))


# ---------- 控制面板 ----------

PAGE = """<!doctype html><meta charset=utf-8><title>切换</title>
<style>
html,body{margin:0;padding:0;min-height:100%%;background:#0b0b0f;color:#e8e8ef;
font:15px/1.6 system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif}
.wrap{max-width:520px;margin:0 auto;padding:48px 20px}
h1{font-size:18px;font-weight:600;letter-spacing:.02em;margin:0 0 24px;color:#a8a8c0}
button{display:block;width:100%%;margin:8px 0;padding:14px 18px;text-align:left;
background:#16161f;color:#e8e8ef;border:1px solid #26263a;border-radius:10px;
font:inherit;cursor:pointer;transition:.15s}
button:hover{background:#1e1e2b;border-color:#3a3a55}
button.on{background:#1e2a3f;border-color:#4a7fd4;color:#9fc4ff}
</style><div class=wrap><h1>当前角色 · %(active)s</h1>%(buttons)s</div>
<script>
async function go(n){await fetch('/switch?name='+encodeURIComponent(n));location.reload()}
</script>"""


def serve_panel(registry: VoiceRegistry, port=8900):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path.startswith("/switch"):
                from urllib.parse import parse_qs, urlparse
                name = parse_qs(urlparse(self.path).query).get("name", [""])[0]
                ok = registry.switch(name)
                self.send_response(200 if ok else 404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": ok}).encode())
                return

            btns = "".join(
                f'<button class="{"on" if k == registry.active else ""}" '
                f"onclick=\"go('{k}')\">{v.get('label', k)}</button>"
                for k, v in registry.characters.items()
            )
            body = (PAGE % {"active": registry.active, "buttons": btns}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)

    srv = HTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    logger.info(f"切换面板: http://127.0.0.1:{port}/")
    return srv
