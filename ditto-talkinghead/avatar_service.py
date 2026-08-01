"""
avatar_service.py  ——  Ditto 真人形象服务 (跑在 Ditto 的 py3.10 venv)

· 启动时加载 StreamSDK。
· POST /generate  : body = 原始 int16 PCM(16k, 单声道) → 用照片生成这句的说话视频(mp4) → 通知播放页。
· POST /generate_filler?index=N : 预生成过渡语视频(缓存进池子, 不推播放, 见"铺垫语"设计)。
· GET  /play_filler?chars=N : 从过渡语池子挑一段推给播放页(不生成, 接近零延迟)；
                              带 chars(预期回复字数) 就按时长占比挑接近的, 不给就纯随机。
· GET  /          : 浏览器播放页(顺序播放视频段 + 身体摆动 + 空闲显示静态照片)。
· GET  /video/<f> : 提供生成的 mp4。
· GET  /events    : SSE, 有新视频段就推给播放页。
· GET  /idle.png  : 静态照片(空闲显示)。
· POST /startup_status?stage=&percent= : 语音管线上报加载进度(内部调用)。
· GET  /startup_status : 播放页轮询, 拿当前加载进度显示进度条。
· GET/POST /settings?stream_sentences=0|1 : 页面上的实时开关(分句流式播放等),
  不用重启管线——语音管线每次处理新回复前会来查一下当前值。

用法: python avatar_service.py --avatar ./avatar/girl.png --port 8902
     python avatar_service.py --backend pytorch   # 需要对比/排障时退回全PyTorch
"""
import argparse, io, json, math, os, random, ssl, threading, time, wave, subprocess
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
# 每句生成的 seg_*(wav/mp4/临时mp4) 从不自动删, 长期挂着会让这个目录无限增长。
# 待命循环(_idle_*/idle_loop.mp4)每次启动固定文件名覆盖写, 不受这套清理影响。
SEGMENT_MAX_AGE_S = 900   # 生成超过15分钟的段清掉(留够时间给SSE重连/慢速播放)
CLEANUP_INTERVAL_S = 300

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
# lmdm(audio2motion扩散模型)编译加速: 同样套 torch.compile, 但用 default 模式(不开
# CUDA Graph) —— reduce-overhead 模式下会跟 rotary embedding 内部张量缓存冲突崩溃,
# 报 "CUDAGraphs 输出被后续运行覆写"。实测(2026-07-30): 在 warp 已编译基础上(19.8fps)
# 再跳到 31.9fps(+61%), 相对纯PyTorch基线(12.5fps) +155%。
_compile_lmdm = True   # 命令行 --no-compile-lmdm 或环境变量 AVATAR_COMPILE_LMDM=0 可关闭
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
_filler_pool = {}                  # index -> 过渡语视频 url("让我想想哈"之类, 预生成好缓存的)
_filler_lock = threading.Lock()
# 语音管线(s2s_pipeline.py)自己上报的加载进度, 供 player.html 轮询显示进度条。
# avatar_service 自己此时早就 ready 了(所以浏览器打得开页面), 管线那边还在加载
# STT/LLM/TTS, WS(8765)连不上, 不然用户完全不知道还要等多久。
_startup_status = {"stage": "等待语音管线启动上报…", "percent": 0}
_startup_lock = threading.Lock()
# 页面上的实时开关(不用重启就能生效): player.html 改了就 POST 到这里存一下,
# 语音管线(lm_output_processor.py)每次处理新回复前来查一下当前值——本机回环
# 请求几毫秒, 比启动时读死一次环境变量灵活, 不用重启整条管线。
_settings = {"stream_sentences": False}
_settings_lock = threading.Lock()


def _cleanup_segments(max_age_s=None):
    """删掉 avatar_out/ 里的 seg_* 生成文件; max_age_s=None 表示不管新旧全删
    (进程刚启动时用, 上一次运行留下的段这次的 _videos/SSE 都不会再引用了)。"""
    now = time.time()
    n = 0
    for fp in OUT_DIR.glob("seg_*"):
        try:
            if max_age_s is None or (now - fp.stat().st_mtime) > max_age_s:
                fp.unlink()
                n += 1
        except OSError:
            pass
    if n:
        print(f"[avatar] 清理了 {n} 个生成文件", flush=True)


def _cleanup_loop():
    while True:
        time.sleep(CLEANUP_INTERVAL_S)
        _cleanup_segments(max_age_s=SEGMENT_MAX_AGE_S)


def load_sdk(avatar_path):
    global _sdk, _avatar
    _avatar = avatar_path
    _cleanup_segments()   # 清掉上一次运行留下的旧段, avatar_out/ 不会无限增长
    threading.Thread(target=_cleanup_loop, daemon=True).start()
    print(f"[avatar] 加载 SDK ...", flush=True)
    _sdk = StreamSDK(CFG_PKL, DATA_ROOT)
    # 人脸注册缓存: 同一照片只注册一次, 后续每句省 ~1s
    _sdk.avatar_registrar = _CachedRegistrar(_sdk.avatar_registrar)
    if _compile_warp:
        print(f"[avatar] warp_network 编译加速(torch.compile, 首次调用较慢, 由下面的待命循环预热吸收)...", flush=True)
        _sdk.warp_f3d.warp_net.model = torch.compile(_sdk.warp_f3d.warp_net.model, mode="reduce-overhead")
    if _compile_lmdm and _sdk.audio2motion.lmdm.model_type == "pytorch":
        # lmdm(audio2motion扩散模型)每句要跑 sampling_timesteps(=50) 步, 每步内部
        # guided_forward 调 2 次 MotionDecoder.forward(CFG), 逐帧 eager 开销明显。
        # 用 default 模式(不开 CUDA Graph): reduce-overhead 会跟 rotary embedding
        # 内部张量缓存冲突崩溃("CUDAGraphs 输出被后续运行覆写"), default 没有该问题。
        print(f"[avatar] lmdm 编译加速(torch.compile, 首次调用较慢, 由下面的待命循环预热吸收)...", flush=True)
        motion_decoder = _sdk.audio2motion.lmdm.model.model
        motion_decoder.forward = torch.compile(motion_decoder.forward, mode="default")
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


def _render(audio_f32: np.ndarray, raw: str, wav: str, out: str):
    """核心渲染: 音频 → 说话视频文件(不涉及命名/队列, 供 _generate/_generate_filler 共用)"""
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


def _generate(audio_f32: np.ndarray) -> str:
    """生成一句回复的说话视频, 直接推给播放页, 返回 mp4 的 url 路径"""
    ts = int(time.time() * 1000)
    raw = str(OUT_DIR / f"seg_{ts}_raw.mp4")
    wav = str(OUT_DIR / f"seg_{ts}.wav")
    out = str(OUT_DIR / f"seg_{ts}.mp4")
    _render(audio_f32, raw, wav, out)
    url = f"/video/seg_{ts}.mp4"
    with _cond:
        _videos.append(url); _cond.notify_all()
    return url


def _generate_filler(audio_f32: np.ndarray, index: int) -> str:
    """预生成一段过渡语("让我想想哈"之类)视频, 固定文件名(不会被 seg_* 的定期清理删掉),
    存进池子(带时长, 供按回复长短挑选合适的过渡语)等 /play_filler 挑, 不直接推播放页。"""
    raw = str(OUT_DIR / f"filler_{index}_raw.mp4")
    wav = str(OUT_DIR / f"filler_{index}.wav")
    out = str(OUT_DIR / f"filler_{index}.mp4")
    _render(audio_f32, raw, wav, out)
    url = f"/video/filler_{index}.mp4"
    duration = len(audio_f32) / 16000
    with _filler_lock:
        _filler_pool[index] = {"url": url, "duration": duration}
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


def serve(port, avatar_path, tls_cert=None, tls_key=None):
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
            if path == "/startup_status":
                with _startup_lock:
                    status = dict(_startup_status)
                return self._bytes(200, "application/json", json.dumps(status).encode())
            if path == "/settings":
                with _settings_lock:
                    settings = dict(_settings)
                return self._bytes(200, "application/json", json.dumps(settings).encode())
            if path == "/play_filler":
                from urllib.parse import parse_qs, urlparse
                qs = parse_qs(urlparse(self.path).query)
                chars = None
                if "chars" in qs:
                    try:
                        chars = float(qs["chars"][0])
                    except ValueError:
                        chars = None
                with _filler_lock:
                    entries = list(_filler_pool.values())
                if not entries:
                    return self._bytes(200, "application/json", b'{"skip":true}')
                if chars is not None and len(entries) > 1:
                    # 回复越长, 后面等的时间大概率越久, 挑一段时长占比相近的过渡语,
                    # 比纯随机更自然(不会出现"啊"一声就秒接超长回复的错位感)
                    entries.sort(key=lambda e: e["duration"])
                    ratio = max(0.0, min(1.0, chars / 80.0))
                    chosen = entries[round(ratio * (len(entries) - 1))]
                else:
                    chosen = random.choice(entries)
                url = chosen["url"]
                with _cond:
                    _videos.append(url); _cond.notify_all()
                return self._bytes(200, "application/json",
                                   json.dumps({"video": url}).encode())
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
            path = self.path.split("?", 1)[0]
            if path == "/startup_status":
                from urllib.parse import parse_qs, urlparse
                qs = parse_qs(urlparse(self.path).query)
                stage = qs.get("stage", [""])[0]
                try:
                    percent = int(qs.get("percent", ["0"])[0])
                except ValueError:
                    percent = 0
                with _startup_lock:
                    _startup_status["stage"] = stage
                    _startup_status["percent"] = max(0, min(100, percent))
                return self._bytes(200, "application/json", b'{"ok":true}')
            if path == "/settings":
                from urllib.parse import parse_qs, urlparse
                qs = parse_qs(urlparse(self.path).query)
                if "stream_sentences" in qs:
                    with _settings_lock:
                        _settings["stream_sentences"] = qs["stream_sentences"][0].lower() not in ("0", "false", "no", "off")
                with _settings_lock:
                    settings = dict(_settings)
                return self._bytes(200, "application/json", json.dumps(settings).encode())
            if path == "/reset_fillers":
                with _filler_lock:
                    _filler_pool.clear()
                return self._bytes(200, "application/json", b'{"ok":true}')
            if path == "/generate_filler":
                from urllib.parse import parse_qs, urlparse
                qs = parse_qs(urlparse(self.path).query)
                try:
                    index = int(qs.get("index", ["0"])[0])
                except ValueError:
                    index = 0
                n = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(n)
                try:
                    audio = _pcm16_to_f32(body)
                    if audio.size < 1600:
                        return self._bytes(200, "application/json", b'{"skip":true}')
                    url = _generate_filler(audio, index)
                    return self._bytes(200, "application/json",
                                       json.dumps({"video": url}).encode())
                except Exception as e:
                    return self._bytes(500, "application/json",
                                       json.dumps({"error": str(e)}).encode())
            if path != "/generate":
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
    scheme = "http"
    if tls_cert and tls_key:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(tls_cert, tls_key)
        srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
        scheme = "https"
    print(f"[avatar] 形象播放页: {scheme}://127.0.0.1:{port}/", flush=True)
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
    ap.add_argument("--no-compile-lmdm", dest="compile_lmdm", action="store_false",
                    help="关闭 lmdm(audio2motion) 的 torch.compile 加速(排障用)")
    ap.set_defaults(compile_lmdm=None)
    ap.add_argument("--tls-cert", default=None, help="TLS 证书路径; 和 --tls-key 一起给才开 HTTPS(网页版局域网访问需要)")
    ap.add_argument("--tls-key", default=None, help="TLS 私钥路径")
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
    # lmdm 编译加速: 命令行 --no-compile-lmdm 优先; 否则看环境变量 AVATAR_COMPILE_LMDM; 默认开
    if args.compile_lmdm is None:
        _compile_lmdm = os.environ.get("AVATAR_COMPILE_LMDM", "1").lower() not in ("0", "false", "no", "off")
    else:
        _compile_lmdm = args.compile_lmdm
    print(f"[avatar] lmdm torch.compile: {'开(首次由预热吸收)' if _compile_lmdm else '关'}", flush=True)
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
    serve(args.port, args.avatar, tls_cert=args.tls_cert, tls_key=args.tls_key)
