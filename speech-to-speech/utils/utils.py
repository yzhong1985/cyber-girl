import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


def report_startup(stage: str, percent: int) -> None:
    """avatar 模式(网页版)下, 把"模型加载到哪一步了"上报给 avatar_service.py 缓存, 供
    player.html 轮询显示进度条——avatar_service.py(8902) 起得比这个管线快得多
    (Ditto自己加载~100s, 管线STT/LLM/TTS加载还要再等几十秒), 页面这段时间是
    打开着但WS连不上的, 不然用户完全不知道还要等多久。失败(没设avatar/网络
    问题)不影响主流程, 纯粹是进度上报。放在这个共享工具模块里(而不是
    s2s_pipeline.py), 是因为 connections/websocket_streamer.py 也要用,
    直接 import s2s_pipeline 在直接执行(__main__)时会变成重复加载整个模块。

    上报目标优先用 AVATAR_PAGE_URL(start_avatar.sh 设的, 固定指向 player.html
    所在的 avatar_service.py, 不管 AVATAR_ENGINE 是 ditto 还是 livetalking 都是
    同一个地址); 没设就退回旧逻辑从 DITTO_AVATAR_URL 派生——AVATAR_ENGINE=
    livetalking 时 DITTO_AVATAR_URL 指向桥接服务(只认 /generate), 不是页面
    本身, 派生出来的地址是错的(实测: 进度上报全部404静默失败, 页面进度条卡
    死在0%), 这就是加 AVATAR_PAGE_URL 的原因。"""
    page_url = os.environ.get("AVATAR_PAGE_URL")
    if page_url:
        base = page_url.rstrip("/")
    else:
        avatar_url = os.environ.get("DITTO_AVATAR_URL")
        if not avatar_url:
            return
        base = avatar_url.rsplit("/generate", 1)[0]
    import ssl
    import urllib.parse
    import urllib.request

    url = f"{base}/startup_status?stage={urllib.parse.quote(stage)}&percent={percent}"
    try:
        ctx = None
        if url.startswith("https://"):
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        urllib.request.urlopen(
            urllib.request.Request(url, data=b"", method="POST"), timeout=3, context=ctx
        ).read()
    except Exception as e:
        logger.debug(f"启动进度上报失败(不致命): {e}")


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def int2float(sound):
    """
    Taken from https://github.com/snakers4/silero-vad
    """

    abs_max = np.abs(sound).max()
    sound = sound.astype("float32")
    if abs_max > 0:
        sound *= 1 / 32768
    sound = sound.squeeze()  # depends on the use case
    return sound
