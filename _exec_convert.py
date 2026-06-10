import subprocess, sys

result = subprocess.run([sys.executable, '-c', '''
import os, json, shutil, random
from pathlib import Path
src = Path(r"c:/Users/admin/Desktop/zhuashengb1/all_data/data")
dst = Path(r"c:/Users/admin/Desktop/zhuashengb1/all_data/yolov8_project/dataset")
for d in ["images/train","images/val","labels/train","labels/val"]:
    (dst/d).mkdir(parents=True, exist_ok=True)
jfiles = list(src.glob("*.json"))
random.seed(42); random.shuffle(jfiles)
split = int(len(jfiles)*0.8); train_j = jfiles[:split]; val_j = jfiles[split:]
for sn,jl in [("train",train_j),("val",val_j)]:
    cnt=0
    for jf in jl:
        imgp=src/(jf.stem+".jpg")
        if not imgp.exists(): continue
        with open(jf,"r",encoding="utf-8") as f: data=json.load(f)
        w,h=data["imageWidth"],data["imageHeight"]
        lines=[]
        for s in data["shapes"]:
            lb=s["label"]
            if lb not in ("hat","nohat"): continue
            cid={"hat":0,"nohat":1}[lb]
            p=s["points"]
            xc=((p[0][0]+p[1][0])/2)/w; yc=((p[0][1]+p[1][1])/2)/h
            bw=abs(p[1][0]-p[0][0])/w; bh=abs(p[1][1]-p[0][1])/h
            lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        if not lines: continue
        shutil.copy(imgp, dst/"images"/sn/imgp.name)
        (dst/"labels"/sn/(jf.stem+".txt")).write_text(chr(10).join(lines))
        cnt+=1
    print(f"{sn}: {cnt}")
(dst/"data.yaml").write_text(f"path: {str(dst.absolute())}\\ntrain: images/train\\nval: images/val\\n\\nc: 2\\nnames: [\"hat\", \"nohat\"]\\n")
print("Conversion complete!")
'''], capture_output=True, text=True)

print(result.stdout)
if result.stderr:
    print(result.stderr)
