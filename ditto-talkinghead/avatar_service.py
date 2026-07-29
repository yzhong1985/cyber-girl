"""
avatar_service.py  ——  Ditto 真人形象服务 (跑在 Ditto 的 py3.10 venv)

· 启动时加载 StreamSDK。
· POST /generate  : body = 原始 int16 PCM(16k, 单声道) → 用照片生成这句的说话视频(mp4) → 通知播放页。
· GET  /          : 浏览器播放页(顺序播放视频段 + 身体摆动 + 空闲显示静态照片)。
· GET  /video/<f> : 提供生成的 mp4。
· GET  /events    : SSE, 有新视频段就推给播放页。
· GET  /idle.png  : 静态照片(空闲显示)。

用法: python avatar_service.py --avatar ./avatar/girl.png --port 8902
     python avatar_service.py --backend pytorch   # 需要对比/排障时退回全PyTorch
"""
import argparse, io, json, math, os, threading, time, wave, subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import cv2
import torch
import librosa
from stream_pipeline_offline import StreamSDK

# 生成后端: pytorch(全PyTorch eager) / onnx(混合: 核心模型走onnxruntime CUDA,
# warp_network 因自定义算子 GridSample3D 仍留 PyTorch)。实测 onnx 快 ~20%(12.5→15fps),
# 画质无感知差异。onnx 版配置由 scripts/build_onnx_hybrid_cfg.py 生成, 已随包保留。
CFG_PKL_BY_BACKEND = {
    "pytorch": "./checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl",
    "onnx": "./checkpoints/ditto_cfg/v0.4_hubert_cfg_onnx.pkl",
}
_backend = "onnx"   # 命令行 --backend 或环境变量 AVATAR_BACKEND 可覆盖
CFG_PKL = CFG_PKL_BY_BACKEND[_backend]
DATA_ROOT = "./checkpoints/ditto_pytorch"
# 用户选定档: 步数50 + 平滑5 + 768
GEN_KWARGS = dict(max_size=768, sampling_timesteps=50, smo_k_d=5)

OUT_DIR = Path("./avatar_out"); OUT_DIR.mkdir(exist_ok=True)

_sdk = None
_rvm = None            # RobustVideoMatting: 抠人像做透明背景
_avatar = None
_gen_lock = threading.Lock()
# warp_network 编译加速: 不管选哪个生成后端, warp_network 因 GridSample3D 算子限制
# 始终是 PyTorch。给它单独套 torch.compile(CUDA Graph 捕获), 消除逐帧 eager 调度开销,
# 与 --backend 正交、独立生效。实测(2026-07-29): 在 onnx 混合后端基础上再 +30%
# (15.0→19.5fps), 换算相对纯PyTorch基线 +56%。代价: 进程内首次调用编译耗时 ~70s,
# 由启动时的待命循环生成(_build_idle_loop)顺带吸收, 用户感知不到。
_compile_warp = True   # 命令行 --no-compile-warp 或环境变量 AVATAR_COMPILE_WARP=0 可关闭
# 抠图开关: True=抠像并合成到背景(慢, 好看); False=直接用照片原背景(快, 省 ~1×音频时长)
# 命令行 --no-matte 或环境变量 AVATAR_MATTE=0 可关闭
_matte = True


class _CachedRegistrar:
    """包住 Ditto 的 avatar_registrar: 同一张照片(相同注册参数)只做一次人脸检测/特征
    提取, 之后直接返回缓存, 省掉每句 ~1s 的重注册。返回时重建顶层 dict 与其中的 list
    容器(浅拷贝), 避免下游对 source_info 的改写污染缓存。"""
    def __init__(self, inner):
        self._inner = inner
        self._cache = {}

    def __call__(self, source_path, **kw):
        key = (source_path, kw.get("max_dim"), kw.get("n_frames"),
               kw.get("crop_scale"), kw.get("crop_vx_ratio"),
               kw.get("crop_vy_ratio"), kw.get("crop_flag_do_rot"))
        info = self._cache.get(key)
        if info is None:
            info = self._inner(source_path, **kw)
            self._cache[key] = info
        return {k: (list(v) if isinstance(v, list) else v) for k, v in info.items()}

    def __getattr__(self, name):
        return getattr(self._inner, name)


_bg_img = None    # 背景图(numpy RGB); None 时用生成的渐变


def _gradient_bg(w, h):
    """暗色柔光渐变(头部区域一点光晕), 衬托人像"""
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    top = np.array([26, 28, 38], np.float32)
    bot = np.array([14, 15, 21], np.float32)
    base = top * (1 - y) + bot * y                       # h x 3
    img = np.repeat(base[:, None, :], w, axis=1)         # h x w x 3
    xx, yy = np.meshgrid(np.linspace(0, 1, w, dtype=np.float32),
                         np.linspace(0, 1, h, dtype=np.float32))
    d = np.sqrt((xx - 0.5) ** 2 + (yy - 0.32) ** 2)
    glow = np.clip(1 - d / 0.6, 0, 1)[..., None] * np.array([28, 30, 42], np.float32)
    return np.clip(img + glow, 0, 255).astype(np.uint8)


def _bg_for(w, h):
    if _bg_img is not None:
        return cv2.resize(_bg_img, (w, h), interpolation=cv2.INTER_AREA)
    return _gradient_bg(w, h)


def _matte_composite(rgb_mp4: str, out_mp4: str, audio_wav: str = None, fps: int = 25):
    """读 Ditto 出的 RGB 视频, 逐帧 RVM 抠像, 合成到背景上, 输出普通 mp4(h264)。"""
    cap = cv2.VideoCapture(rgb_mp4)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    bg = torch.from_numpy(_bg_for(w, h)).float().div(255).permute(2, 0, 1).unsqueeze(0).cuda()
    cmd = ["ffmpeg", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-"]
    if audio_wav:
        cmd += ["-i", audio_wav]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast"]
    if audio_wav:
        cmd += ["-c:a", "aac", "-shortest"]
    cmd += [out_mp4]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    rec = [None] * 4
    with torch.no_grad(), torch.autocast("cuda"):
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            s = torch.from_numpy(rgb).float().div(255).permute(2, 0, 1).unsqueeze(0).cuda()
            fgr, pha, *rec = _rvm(s, *rec, downsample_ratio=0.4)
            comp = fgr * pha + bg * (1 - pha)
            out = comp[0].permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()
            proc.stdin.write(out.tobytes())
    cap.release()
    proc.stdin.close()
    proc.wait()


def _mux_audio(rgb_mp4: str, out_mp4: str, audio_wav: str = None):
    """不抠图时的快路径: 直接把 Ditto 出的视频(照片原背景)封成 mp4, 只需合入音频,
    视频流 copy 不重编码, 几乎零耗时。audio_wav 为空则只是转封装。
    注意: 不能用 -shortest —— 视频(ceil(时长*25帧)/25s)总是比音频略长(至多约40ms),
    -shortest 配合 -c:v copy(stream copy 无法精确到帧, 只能在关键帧边界裁切)会一次
    过量裁掉好几个尾帧(实测最多丢4帧), 而不是预期的零点几帧。不设它, 让视频保留完整
    帧数, 音频结尾静音(数十毫秒, 无感知), 比丢真实动作帧好得多。"""
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-i", rgb_mp4]
    if audio_wav:
        cmd += ["-i", audio_wav, "-c:v", "copy", "-c:a", "aac"]
    else:
        cmd += ["-c:v", "copy", "-an"]
    cmd += [out_mp4]
    subprocess.run(cmd, check=False)


_videos = []                       # 已生成的视频 url 列表(供 SSE 顺序下发)
_cond = threading.Condition()


def load_sdk(avatar_path):
    global _sdk, _avatar
    _avatar = avatar_path
    print(f"[avatar] 加载 SDK ...", flush=True)
    _sdk = StreamSDK(CFG_PKL, DATA_ROOT)
    # 人脸注册缓存: 同一照片只注册一次, 后续每句省 ~1s
    _sdk.avatar_registrar = _CachedRegistrar(_sdk.avatar_registrar)
    if _compile_warp:
        print(f"[avatar] warp_network 编译加速(torch.compile, 首次调用较慢, 由下面的待命循环预热吸收)...", flush=True)
        _sdk.warp_f3d.warp_net.model = torch.compile(_sdk.warp_f3d.warp_net.model, mode="reduce-overhead")
    global _rvm
    if _matte:
        print(f"[avatar] 加载 RVM 抠像模型 ...", flush=True)
        _rvm = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3", trust_repo=True).cuda().eval()
    else:
        print(f"[avatar] 抠图已关闭(--no-matte), 跳过 RVM 加载, 用照片原背景", flush=True)
    # 生成当前形象的待命循环(乒乓无缝) + 预热 + 人脸检测确认
    print(f"[avatar] 生成待命循环 ...", flush=True)
    _build_idle_loop()
    print(f"[avatar] 就绪, 形象 = {avatar_path}", flush=True)


def _pcm16_to_f32(body: bytes) -> np.ndarray:
    return np.frombuffer(body, dtype=np.int16).astype(np.float32) / 32768.0


def _generate(audio_f32: np.ndarray) -> str:
    """生成一句的说话视频, 返回 mp4 的 url 路径"""
    ts = int(time.time() * 1000)
    raw = str(OUT_DIR / f"seg_{ts}_raw.mp4")
    wav = str(OUT_DIR / f"seg_{ts}.wav")
    out = str(OUT_DIR / f"seg_{ts}.mp4")

    # 落一个 wav 供合流
    with wave.open(wav, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes((np.clip(audio_f32, -1, 1) * 32767).astype(np.int16).tobytes())

    num_f = math.ceil(len(audio_f32) / 16000 * 25)
    with _gen_lock:
        _sdk.setup(_avatar, raw, **GEN_KWARGS)
        _sdk.setup_Nd(N_d=num_f)
        _sdk.audio2motion_queue.put(_sdk.wav2feat.wav2feat(audio_f32))
        _sdk.close()
        tmp_v = _sdk.tmp_output_path
        if _matte:
            # 抠像 → 合成到背景 → mp4(带音频)
            _matte_composite(tmp_v, out, audio_wav=wav)
        else:
            # 快路径: 用照片原背景, 只合入音频(视频流 copy 不重编码)
            _mux_audio(tmp_v, out, audio_wav=wav)
    url = f"/video/seg_{ts}.mp4"
    with _cond:
        _videos.append(url); _cond.notify_all()
    return url


def _build_idle_loop(seconds=5):
    """用静音生成一段自带眨眼/微动的待命视频, 做成'正放+倒放'乒乓 → 无缝循环。
    同时充当预热 + 人脸检测确认。每次服务启动(换照片)都会重建, 自动匹配当前形象。"""
    silence = np.zeros(16000 * seconds, dtype=np.float32)
    raw = str(OUT_DIR / "_idle_raw.mp4")
    fwd = str(OUT_DIR / "_idle_fwd.mp4")
    out = str(OUT_DIR / "idle_loop.mp4")
    num_f = math.ceil(len(silence) / 16000 * 25)
    try:
        with _gen_lock:
            _sdk.setup(_avatar, raw, **GEN_KWARGS)
            _sdk.setup_Nd(N_d=num_f)
            _sdk.audio2motion_queue.put(_sdk.wav2feat.wav2feat(silence))
            _sdk.close()
            tmp_v = _sdk.tmp_output_path
            if _matte:
                # 抠像 → 合成到背景 → mp4(无音频)
                _matte_composite(tmp_v, fwd, audio_wav=None)
            else:
                # 不抠图: 待命循环也用照片原背景, 与回复保持一致
                _mux_audio(tmp_v, fwd, audio_wav=None)
        # 乒乓拼接(正放+倒放) → loop 播放天衣无缝
        subprocess.run(
            f'ffmpeg -loglevel error -y -i "{fwd}" '
            f'-filter_complex "[0:v]split[a][b];[b]reverse[r];[a][r]concat=n=2:v=1[out]" '
            f'-map "[out]" -c:v libx264 -pix_fmt yuv420p -an "{out}"',
            shell=True, check=False)
        print(f"[avatar] 待命循环(乒乓, 已合成背景)已生成: {out}", flush=True)
    except Exception as e:
        print(f"[avatar] 待命循环生成失败(不致命): {e}", flush=True)


PAGE = """<!doctype html><html lang=zh><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>cyber girl</title>
<style>
 html,body{margin:0;height:100%;background:#0b0b0f;overflow:hidden;font-family:system-ui,"Microsoft YaHei",sans-serif}
 #stage{position:fixed;inset:0;display:flex;align-items:center;justify-content:center}
 /* 身体摆动: 从底部为支点轻轻摇摆 + 呼吸 */
 @keyframes sway{0%,100%{transform:rotate(-1.3deg) translateX(-0.5%) scale(1.0)}
                 50%{transform:rotate(1.3deg) translateX(0.5%) scale(1.016)}}
 .avatar{height:100vh;max-width:100vw;object-fit:contain;transform-origin:50% 92%;
         animation:sway var(--sway,5s) ease-in-out infinite;will-change:transform}
 #vid{display:none}
 #hud{position:fixed;left:12px;top:10px;color:#7a7a90;font-size:13px}
 #hud b{color:#9fc4ff}
 #gate{position:fixed;inset:0;background:#0b0b0fcc;display:flex;align-items:center;justify-content:center;cursor:pointer}
 #gate div{color:#e8e8ef;font-size:18px;padding:14px 26px;border:1px solid #33334a;border-radius:12px;background:#16161f}
</style></head><body>
<div id=stage>
  <img id=idle class=avatar src="/idle.png">
  <video id=vid class=avatar playsinline></video>
</div>
<div id=hud>cyber girl · <b id=st>待命</b></div>
<div id=gate><div>▶ 点击进入</div></div>
<script>
 const vid=document.getElementById('vid'), idle=document.getElementById('idle'),
       st=document.getElementById('st'), gate=document.getElementById('gate');
 const q=[]; let playing=false, ready=false;
 gate.onclick=()=>{ready=true; gate.style.display='none'; vid.muted=false; pump();};
 function showIdle(){vid.style.display='none'; idle.style.display='block'; st.textContent='待命';}
 function pump(){
   if(playing||!ready) return;
   const next=q.shift();
   if(!next){showIdle(); return;}
   playing=true; st.textContent='说话中';
   idle.style.display='none'; vid.style.display='block';
   vid.src=next; vid.play().catch(e=>{playing=false;});
 }
 vid.onended=()=>{playing=false; pump();};
 vid.onerror=()=>{playing=false; pump();};
 const es=new EventSource('/events');
 es.onmessage=e=>{try{const d=JSON.parse(e.data); if(d.video){q.push(d.video); pump();}}catch(_){}};
</script></body></html>"""


def serve(port, avatar_path):
    load_sdk(avatar_path)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass

        def _bytes(self, code, ctype, data, extra=None):
            self.send_response(code); self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            if extra:
                for k, v in extra.items(): self.send_header(k, v)
            self.end_headers(); self.wfile.write(data)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", ""):
                html = Path("player.html").read_bytes()   # 实时读文件, 改样式不用重启
                return self._bytes(200, "text/html; charset=utf-8", html)
            if path == "/idle.png":
                return self._bytes(200, "image/png", Path(_avatar).read_bytes())
            if path.startswith("/video/"):
                fp = (OUT_DIR / Path(path).name).resolve()
                if fp.is_file() and str(fp).startswith(str(OUT_DIR.resolve())):
                    ctype = "video/webm" if fp.suffix == ".webm" else "video/mp4"
                    return self._bytes(200, ctype, fp.read_bytes(),
                                       {"Cache-Control": "no-store"})
                self.send_response(404); self.end_headers(); return
            if path == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                idx = len(_videos)   # 只推此刻之后的新视频
                try:
                    while True:
                        with _cond:
                            while idx >= len(_videos):
                                _cond.wait(timeout=15)
                                if idx >= len(_videos):
                                    self.wfile.write(b": ping\n\n"); self.wfile.flush()
                            url = _videos[idx]; idx += 1
                        self.wfile.write(f"data: {json.dumps({'video': url})}\n\n".encode())
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
            self.send_response(404); self.end_headers()

        def do_POST(self):
            if self.path.split("?", 1)[0] != "/generate":
                self.send_response(404); self.end_headers(); return
            n = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n)
            try:
                audio = _pcm16_to_f32(body)
                if audio.size < 1600:   # <0.1s, 忽略
                    return self._bytes(200, "application/json", b'{"skip":true}')
                url = _generate(audio)
                dur = round(len(audio) / 16000, 2)
                return self._bytes(200, "application/json",
                                   json.dumps({"video": url, "duration": dur}).encode())
            except Exception as e:
                return self._bytes(500, "application/json",
                                   json.dumps({"error": str(e)}).encode())

    srv = ThreadingHTTPServer(("0.0.0.0", port), H)
    srv.daemon_threads = True
    print(f"[avatar] 形象播放页: http://127.0.0.1:{port}/", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--avatar", default="./avatar/girl.png")
    ap.add_argument("--bg", default=None, help="背景图路径; 不给则用暗色渐变")
    ap.add_argument("--port", type=int, default=8902)
    ap.add_argument("--no-matte", dest="matte", action="store_false",
                    help="关闭抠图: 用照片原背景, 每句省 ~1×音频时长(更快)")
    ap.set_defaults(matte=None)
    ap.add_argument("--backend", choices=["pytorch", "onnx"], default=None,
                    help="生成后端: onnx(默认, 混合onnxruntime, 快~20%%) / pytorch(全PyTorch)")
    ap.add_argument("--no-compile-warp", dest="compile_warp", action="store_false",
                    help="关闭 warp_network 的 torch.compile 加速(排障用)")
    ap.set_defaults(compile_warp=None)
    args = ap.parse_args()
    # 抠图开关: 命令行 --no-matte 优先; 否则看环境变量 AVATAR_MATTE(0/false 关); 默认开
    if args.matte is None:
        _matte = os.environ.get("AVATAR_MATTE", "1").lower() not in ("0", "false", "no", "off")
    else:
        _matte = args.matte
    print(f"[avatar] 抠图: {'开(合成到背景)' if _matte else '关(照片原背景, 更快)'}", flush=True)
    # 生成后端: 命令行 --backend 优先; 否则看环境变量 AVATAR_BACKEND; 默认 onnx
    _backend = args.backend or os.environ.get("AVATAR_BACKEND", "onnx")
    CFG_PKL = CFG_PKL_BY_BACKEND[_backend]
    print(f"[avatar] 生成后端: {_backend}"
          + ("(混合onnxruntime, 快~20%)" if _backend == "onnx" else "(全PyTorch)"), flush=True)
    # warp_network 编译加速: 命令行 --no-compile-warp 优先; 否则看环境变量 AVATAR_COMPILE_WARP; 默认开
    if args.compile_warp is None:
        _compile_warp = os.environ.get("AVATAR_COMPILE_WARP", "1").lower() not in ("0", "false", "no", "off")
    else:
        _compile_warp = args.compile_warp
    print(f"[avatar] warp_network torch.compile: {'开(+约30%, 首次约70s由预热吸收)' if _compile_warp else '关'}", flush=True)
    # 背景: 命令行 --bg 优先, 否则自动找 avatar/background.*, 都没有就用渐变
    bg_path = args.bg
    if not bg_path:
        for cand in ("avatar/background.png", "avatar/background.jpg"):
            if os.path.exists(cand):
                bg_path = cand; break
    if bg_path and os.path.exists(bg_path):
        _bg_img = cv2.cvtColor(cv2.imread(bg_path), cv2.COLOR_BGR2RGB)
        print(f"[avatar] 背景图: {bg_path}", flush=True)
    else:
        print("[avatar] 背景: 暗色渐变(默认)", flush=True)
    serve(args.port, args.avatar)
