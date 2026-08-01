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

import base64
import json
import logging
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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

    def current_ref_text(self):
        """参考音频对应的文字稿；faster-qwen3-tts 声音克隆需要它"""
        with self.lock:
            return self.characters[self.active].get("ref_text", "")

    def current_prompt(self):
        with self.lock:
            return self.characters[self.active]["system_prompt"]

    _DEFAULT_FILLERS = ["嗯，让我想想", "让我想想哈", "等一下"]

    def current_fillers(self):
        """回复前的过渡语候选(贴合角色语气); characters.json 没配就用通用兜底"""
        with self.lock:
            return self.characters[self.active].get("fillers") or list(self._DEFAULT_FILLERS)

    def current_avatar_photo(self):
        """角色绑定的照片(Ditto形象); 没配的话返回 None, 调用方兜底用启动时的 AVATAR_PHOTO"""
        with self.lock:
            return self.characters[self.active].get("avatar_photo")

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

    # ---------- 增删(多角色管理面板用) ----------

    def save_config(self):
        """把当前 self.characters 写回 characters.json(新建/删除角色都要落盘,
        不然重启进程就丢了)"""
        with self.lock:
            data = json.dumps(self.characters, ensure_ascii=False, indent=2)
        self.config_path.write_text(data, encoding="utf-8")

    def add_character(self, name: str, data: dict):
        """新建的角色一律标 builtin=False(能被删), 内置角色是手动在
        characters.json 里标 builtin=true 的, 这里加不了那个标记。"""
        with self.lock:
            entry = dict(data)
            entry["builtin"] = False
            self.characters[name] = entry
        self.save_config()

    def delete_character(self, name: str):
        """返回 (成功?, 出错信息)。内置角色/仅剩最后一个角色不让删；删的正好
        是当前角色时, 落盘后再切到一个兜底角色(优先内置的), 让 TTS/Ditto/LLM
        那些绑定在 switch() 回调上的热切换正常触发, 不会留在一个已经被删掉的
        角色上。"""
        with self.lock:
            if name not in self.characters:
                return False, "角色不存在"
            if self.characters[name].get("builtin"):
                return False, "内置角色不能删除"
            if len(self.characters) <= 1:
                return False, "至少要保留一个角色"
            was_active = (self.active == name)
            del self.characters[name]
            fallback = None
            if was_active:
                fallback = next(
                    (k for k, v in self.characters.items() if v.get("builtin")),
                    next(iter(self.characters)),
                )
        self.save_config()
        if fallback:
            self.switch(fallback)   # 锁外调用: switch()自己加锁+触发回调(换音色/头像/人格)
        return True, ""


# ---------- 控制面板 ----------

PAGE = """<!doctype html><meta charset=utf-8><title>角色管理</title>
<style>
html,body{margin:0;padding:0;min-height:100%%;background:#0b0b0f;color:#e8e8ef;
font:15px/1.6 system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif}
.wrap{max-width:560px;margin:0 auto;padding:48px 20px 80px}
h1{font-size:18px;font-weight:600;letter-spacing:.02em;margin:0 0 24px;color:#a8a8c0}
h2{font-size:14px;font-weight:600;letter-spacing:.02em;margin:40px 0 12px;color:#a8a8c0}
button{display:block;width:100%%;margin:8px 0;padding:14px 18px;text-align:left;
background:#16161f;color:#e8e8ef;border:1px solid #26263a;border-radius:10px;
font:inherit;cursor:pointer;transition:.15s}
button:hover{background:#1e1e2b;border-color:#3a3a55}
button.on{background:#1e2a3f;border-color:#4a7fd4;color:#9fc4ff}
.card{display:flex;align-items:center;gap:8px;margin:8px 0}
.card .name{flex:1;margin:0}
.card .del{background:none;border:1px solid #3a2020;color:#e08a8a;padding:8px 12px;
margin:0;width:auto;cursor:pointer;font-size:13px;border-radius:8px}
.card .del:hover{background:#2a1616}
form{background:#16161f;border:1px solid #26263a;border-radius:10px;padding:16px}
label{display:block;font-size:13px;color:#a8a8c0;margin:14px 0 4px}
label:first-child{margin-top:0}
input[type=text],textarea{width:100%%;box-sizing:border-box;background:#0b0b0f;color:#e8e8ef;
border:1px solid #26263a;border-radius:8px;padding:10px;font:inherit}
textarea{min-height:100px;resize:vertical}
input[type=file]{width:100%%;color:#a8a8c0;font-size:13px}
.confirm{display:flex;align-items:flex-start;gap:8px;margin-top:16px;font-size:13px;color:#c8c8d8}
.confirm input{margin-top:3px;width:auto}
.submit{margin-top:16px;background:#1e2a3f;border-color:#4a7fd4;color:#9fc4ff;text-align:center;cursor:pointer}
.submit:hover{background:#243452}
.msg{font-size:13px;margin-top:10px;min-height:18px}
.msg.err{color:#e08a8a}
</style><div class=wrap>
<h1>当前角色 · %(active)s</h1>
%(cards)s
<h2>新建角色</h2>
<form onsubmit="return false">
  <label>角色名(英文/数字/下划线, 作为唯一标识, 建完不能改)</label>
  <input type=text id=name placeholder="例如 mika">
  <label>显示名</label>
  <input type=text id=label placeholder="例如 米卡 · 元气向">
  <label>形象照片</label>
  <input type=file id=photo accept="image/*">
  <label>参考语音(wav, 5~15秒清晰人声, 决定音色)</label>
  <input type=file id=voice accept="audio/wav,.wav">
  <label>参考语音对应文字(留空则自动转写, 转写要几十秒)</label>
  <input type=text id=reftext placeholder="留空自动转写">
  <label>性格设定(system prompt)</label>
  <textarea id=prompt placeholder="你叫XX, 性格是..."></textarea>
  <div class=confirm>
    <input type=checkbox id=rights>
    <span>我确认已获得该照片/声音使用者的授权(本人或已授权素材), 不用于冒充真实公众人物</span>
  </div>
  <button class=submit onclick="create()">创建并自动切换</button>
  <div class=msg id=msg></div>
</form>
</div>
<script>
async function go(n){await fetch('/switch?name='+encodeURIComponent(n));location.reload()}
async function del(n){
  if(!confirm('删除角色「'+n+'」？此操作不可撤销。')) return;
  const r=await fetch('/delete?name='+encodeURIComponent(n));
  const j=await r.json();
  if(!j.ok){ alert(j.error||'删除失败'); return; }
  location.reload();
}
function readFile(f){
  return new Promise((res,rej)=>{
    if(!f){res(null);return}
    const r=new FileReader();
    r.onload=()=>res(r.result.split(',')[1]);
    r.onerror=rej;
    r.readAsDataURL(f);
  });
}
async function create(){
  const msg=document.getElementById('msg');
  msg.className='msg'; msg.textContent='';
  const name=document.getElementById('name').value.trim();
  const label=document.getElementById('label').value.trim();
  const photoFile=document.getElementById('photo').files[0];
  const voiceFile=document.getElementById('voice').files[0];
  const reftext=document.getElementById('reftext').value.trim();
  const prompt=document.getElementById('prompt').value.trim();
  const rights=document.getElementById('rights').checked;
  if(!name||!label||!photoFile||!voiceFile||!prompt){
    msg.className='msg err'; msg.textContent='角色名/显示名/照片/语音/性格设定都要填'; return;
  }
  if(!rights){
    msg.className='msg err'; msg.textContent='请先勾选授权确认'; return;
  }
  msg.textContent='创建中(自动转写语音可能要几十秒, 别刷新)...';
  const photo_b64=await readFile(photoFile);
  const voice_b64=await readFile(voiceFile);
  const body={
    name, label, prompt, reftext, rights,
    photo_b64, photo_ext: (photoFile.name.split('.').pop()||'png'),
    voice_b64, voice_ext: (voiceFile.name.split('.').pop()||'wav'),
  };
  try{
    const r=await fetch('/create',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const j=await r.json();
    if(!j.ok){ msg.className='msg err'; msg.textContent=j.error||'创建失败'; return; }
    location.reload();
  }catch(e){ msg.className='msg err'; msg.textContent='请求失败: '+e; }
}
</script>"""

_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,32}$")


def serve_panel(registry: VoiceRegistry, port=8900):
    # voice_registry.py 本身就躺在 speech-to-speech/ 里, 用 __file__ 定位比
    # 依赖进程 cwd(config_path 常是相对路径 "characters.json") 更可靠。
    sts_dir = Path(__file__).resolve().parent
    project_root = sts_dir.parent
    ditto_avatar_dir = project_root / "ditto-talkinghead" / "avatar"
    voices_dir = sts_dir / "voices"

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            from urllib.parse import parse_qs, urlparse

            if self.path.startswith("/switch"):
                name = parse_qs(urlparse(self.path).query).get("name", [""])[0]
                ok = registry.switch(name)
                return self._json(200 if ok else 404, {"ok": ok})

            if self.path.startswith("/delete"):
                name = parse_qs(urlparse(self.path).query).get("name", [""])[0]
                ok, err = registry.delete_character(name)
                return self._json(200 if ok else 400, {"ok": ok, "error": err})

            cards = "".join(
                '<div class=card>'
                f'<button class="name {"on" if k == registry.active else ""}" '
                f"onclick=\"go('{k}')\">{v.get('label', k)}</button>"
                + ("" if v.get("builtin") else f"<button class=del onclick=\"del('{k}')\">删除</button>")
                + "</div>"
                for k, v in registry.characters.items()
            )
            body = (PAGE % {"active": registry.active, "cards": cards}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if not self.path.startswith("/create"):
                return self._json(404, {"ok": False, "error": "not found"})
            try:
                length = int(self.headers.get("Content-Length", 0))
                data = json.loads(self.rfile.read(length).decode("utf-8"))
            except Exception as e:
                return self._json(400, {"ok": False, "error": f"请求体解析失败: {e}"})

            name = (data.get("name") or "").strip()
            label = (data.get("label") or "").strip()
            prompt = (data.get("prompt") or "").strip()
            photo_b64 = data.get("photo_b64")
            voice_b64 = data.get("voice_b64")
            photo_ext = re.sub(r"[^a-zA-Z0-9]", "", (data.get("photo_ext") or "png"))[:8] or "png"
            voice_ext = re.sub(r"[^a-zA-Z0-9]", "", (data.get("voice_ext") or "wav"))[:8] or "wav"
            ref_text_override = (data.get("reftext") or "").strip()

            if not data.get("rights"):
                return self._json(400, {"ok": False, "error": "缺少授权确认"})
            if not _NAME_RE.match(name):
                return self._json(400, {"ok": False, "error": "角色名只能是字母/数字/下划线/短横线, 1~32位"})
            if name in registry.characters:
                return self._json(400, {"ok": False, "error": "角色名已存在"})
            if not label or not prompt or not photo_b64 or not voice_b64:
                return self._json(400, {"ok": False, "error": "显示名/照片/语音/性格设定都不能为空"})

            try:
                photo_bytes = base64.b64decode(photo_b64)
                voice_bytes = base64.b64decode(voice_b64)
            except Exception as e:
                return self._json(400, {"ok": False, "error": f"文件解码失败: {e}"})

            ditto_avatar_dir.mkdir(parents=True, exist_ok=True)
            photo_rel = f"avatar/user_{name}.{photo_ext}"
            (ditto_avatar_dir / f"user_{name}.{photo_ext}").write_bytes(photo_bytes)

            voices_dir.mkdir(parents=True, exist_ok=True)
            voice_rel = f"voices/user_{name}.{voice_ext}"
            voice_path = voices_dir / f"user_{name}.{voice_ext}"
            voice_path.write_bytes(voice_bytes)

            ref_text = ref_text_override
            if not ref_text:
                try:
                    from STT.transcribe_ref import transcribe_ref_audio
                    ref_text = transcribe_ref_audio(str(voice_path))
                except Exception as e:
                    logger.warning(
                        f"[面板] 参考语音自动转写失败(角色仍会创建, 但ref_text为空, "
                        f"声音克隆效果会变差, 可以之后手动编辑characters.json补上): {e}"
                    )
                    ref_text = ""

            registry.add_character(name, {
                "label": label,
                "avatar_photo": photo_rel,
                "ref_audio": voice_rel,
                "ref_text": ref_text,
                "llm_model": registry.characters[registry.active].get("llm_model", "default"),
                "system_prompt": prompt,
            })
            registry.switch(name)
            return self._json(200, {"ok": True})

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    logger.info(f"角色管理面板: http://127.0.0.1:{port}/")
    return srv
