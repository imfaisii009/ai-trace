# Score test sets with the shipped weights and fine-tuned checkpoints, the way the page does (mean sigmoid over the 384 grid).
# Usage: eval_local.py <sets.json> <out.json> [ckpt.pt ...]   sets.json: {name: {"label": 0|1, "files": [...]}}
import json, sys, numpy as np, torch, timm
from PIL import Image, ImageOps
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
N = 384; MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
def starts(L):
    s = list(range(0, L - N + 1, N)) or [0]
    if s[-1] != max(L - N, 0): s.append(L - N)
    return s
def tile(a, y0, x0):
    H, W, _ = a.shape
    return a[np.minimum(np.arange(y0, y0 + N), H - 1)][:, np.minimum(np.arange(x0, x0 + N), W - 1)]
def tiles(path):
    im = ImageOps.exif_transpose(Image.open(path)).convert('RGB'); k = min(1, 3072 / max(im.size))
    if k < 1: im = im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS)
    a = np.asarray(im); H, W, _ = a.shape
    return torch.from_numpy(np.stack([(tile(a, y, x) / 255.0 - MEAN) / STD for y in starts(H) for x in starts(W)]).astype(np.float32).transpose(0, 3, 1, 2))
def load(ck):
    m = timm.create_model('vit_small_patch16_384', pretrained=False, num_classes=1)
    sd = {k.removeprefix('vit.'): v for k, v in load_file(hf_hub_download('OwensLab/commfor-model-384', 'model.safetensors')).items()} if ck == 'base' else torch.load(ck, map_location='cpu')
    m.load_state_dict(sd); return m.eval().to(dev)
sets = json.load(open(sys.argv[1])); models = ['base'] + sys.argv[3:]
cache = {}
out = {}
for ck in models:
    m = load(ck); res = {}
    with torch.no_grad():
        for name, s in sets.items():
            r = []
            for f in s['files']:
                t = cache.get(f)
                if t is None: t = cache[f] = tiles(f)
                p = torch.cat([torch.sigmoid(m(t[i:i + 32].to(dev))).float().cpu() for i in range(0, len(t), 32)]).flatten().numpy()
                r.append({'file': f, 'score': float(p.mean()), 'hot': float((p >= 0.5).mean())})
            res[name] = r
    out[ck] = res; print('scored', ck, flush=True)
json.dump(out, open(sys.argv[2], 'w'))
def auroc(pos, neg):
    pos, neg = np.array(pos), np.array(neg)
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / max(1, len(pos) * len(neg)))
reals = [n for n, s in sets.items() if s['label'] == 0]; fakes = [n for n, s in sets.items() if s['label'] == 1]
for ck, res in out.items():
    print(f'\n== {ck}')
    allreal = [x['score'] for n in reals for x in res[n]]
    for n in reals + fakes:
        sc = np.array([x['score'] for x in res[n]]); hot = np.mean([x['hot'] for x in res[n]])
        extra = f"  AUROC vs all reals {auroc(sc, allreal):.3f}" if n in fakes else ''
        print(f"{n:22s} n={len(sc):4d}  flagged@0.2 {np.mean(sc >= 0.2):.2f}  median {np.median(sc):.3f}  hot tiles {hot:.2f}{extra}")
