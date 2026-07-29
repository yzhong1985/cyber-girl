import torch, cv2, numpy as np, time
print("加载 RVM ...", flush=True)
model = torch.hub.load("PeterL1n/RobustVideoMatting","mobilenetv3",trust_repo=True).cuda().eval()
src_mp4="avatar_out/idle_loop.mp4"
cap=cv2.VideoCapture(src_mp4); rec=[None]*4
bg=torch.tensor([0.15,0.85,0.15]).view(1,3,1,1).cuda()  # 绿幕预览
frames=[]; t=time.time()
with torch.no_grad(), torch.autocast("cuda"):
  while True:
    ok,fr=cap.read()
    if not ok: break
    rgb=cv2.cvtColor(fr,cv2.COLOR_BGR2RGB)
    s=torch.from_numpy(rgb).float().div(255).permute(2,0,1).unsqueeze(0).cuda()
    fgr,pha,*rec=model(s,*rec,downsample_ratio=0.4)
    comp=fgr*pha+bg*(1-pha)
    frames.append(cv2.cvtColor((comp[0].permute(1,2,0).clamp(0,1).cpu().numpy()*255).astype(np.uint8),cv2.COLOR_RGB2BGR))
cap.release()
n=len(frames); dt=time.time()-t
h,w=frames[0].shape[:2]
vw=cv2.VideoWriter("avatar_out/_matte_preview.mp4",cv2.VideoWriter_fourcc(*'mp4v'),25,(w,h))
for f in frames: vw.write(f)
vw.release()
print(f"抠像完成 {n}帧/{dt:.1f}s = {n/dt:.1f}fps, {w}x{h} -> avatar_out/_matte_preview.mp4  显存+{torch.cuda.max_memory_allocated()/1e9:.2f}G", flush=True)
