# Fine-tune Community Forensics ViT-S/16 384 on phone photos vs Grok edits of them, with replay of generic generated images.
# Both classes get the same random resampling and JPEG re-encoding, so neither size nor compression gives the label away.
import io, json, math, random, sys, time, os
import numpy as np, torch, timm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
D, OUT = sys.argv[1], sys.argv[2]
EPOCHS = int(os.environ.get('EPOCHS', 8)); BS = int(os.environ.get('BS', 64)); LR = float(os.environ.get('LR', 3e-5)); LIMIT = int(os.environ.get('LIMIT', 0)); NW = int(os.environ.get('NW', min(32, os.cpu_count())))  # NW=0 on macOS (spawn re-runs this script)
N = 384; MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
dev = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
items = json.load(open(f'{D}/manifest.json'))
if LIMIT: random.seed(0); random.shuffle(items); items = items[:LIMIT]
os.makedirs(OUT, exist_ok=True)
def tile(a, y0, x0):
    H, W, _ = a.shape
    return a[np.minimum(np.arange(y0, y0 + N), H - 1)][:, np.minimum(np.arange(x0, x0 + N), W - 1)]
def shrink(im, k):
    return im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), random.choice([Image.LANCZOS, Image.BICUBIC, Image.BILINEAR])) if k < 1 else im
class Train(Dataset):
    REPEAT = {'crop': 1, 'master': 4, 'edit': 8, 'replay': 4, 'replay_real': 4}
    def __init__(self, items): self.items = [it for it in items if it['split'] == 'train' for _ in range(self.REPEAT[it['kind']])]
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        it = self.items[i]; im = Image.open(it['path']).convert('RGB')
        if it['kind'] == 'master': im = shrink(im, random.uniform(0.4, 1.0))
        elif it['kind'] != 'crop' and random.random() < 0.5: im = shrink(im, random.uniform(0.6, 1.0))
        a = np.asarray(im); H, W, _ = a.shape
        a = tile(a, random.randint(0, max(0, H - N)), random.randint(0, max(0, W - N)))
        im = Image.fromarray(a)
        if random.random() < 0.6:
            b = io.BytesIO(); im.save(b, 'JPEG', quality=random.randint(70, 98)); im = Image.open(b).convert('RGB')
        if random.random() < 0.5: im = im.transpose(Image.FLIP_LEFT_RIGHT)
        x = (np.asarray(im, np.float32) / 255 - MEAN) / STD
        return torch.from_numpy(x.transpose(2, 0, 1).copy()), torch.tensor([float(it['label'])])
def starts(L):
    s = list(range(0, L - N + 1, N)) or [0]
    if s[-1] != max(L - N, 0): s.append(L - N)
    return s
@torch.no_grad()
def page_score(model, im):
    k = min(1, 3072 / max(im.size)); im = shrink(im, k) if k < 1 else im
    a = np.asarray(im.convert('RGB')); H, W, _ = a.shape
    t = np.stack([(tile(a, y, x) / 255.0 - MEAN) / STD for y in starts(H) for x in starts(W)]).astype(np.float32).transpose(0, 3, 1, 2)
    with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == 'cuda'):
        return torch.sigmoid(model(torch.from_numpy(t).to(dev)).float()).mean().item()
def auroc(pos, neg):
    pos, neg = np.array(pos), np.array(neg)
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / max(1, len(pos) * len(neg)))
def evaluate(model):
    model.eval(); v = [it for it in items if it['split'] == 'val']
    edits = {it['group']: [] for it in v if it['kind'] == 'edit'}
    for it in v:
        if it['kind'] == 'edit': edits[it['group']].append(page_score(model, Image.open(it['path'])))
    sizes = {it['group']: Image.open(it['path']).size for it in v if it['kind'] == 'edit'}
    real_native, real_matched = [], []
    for it in v:
        if it['kind'] != 'master': continue
        real_native.append(page_score(model, Image.open(it['orig'])))
        if it['group'] in sizes:
            m = Image.open(it['path']).convert('RGB'); w, h = sizes[it['group']]; s = max(w, h) / max(m.size)
            real_matched.append(page_score(model, m.resize((round(m.width * s), round(m.height * s)), Image.LANCZOS)))
    replay = [page_score(model, Image.open(it['path'])) for it in v if it['kind'] == 'replay']
    e = [s for ss in edits.values() for s in ss]
    rr = [page_score(model, Image.open(it['path'])) for it in v if it['kind'] == 'replay_real']
    r = {'edits_vs_matched_real': auroc(e, real_matched), 'edits_vs_native_real': auroc(e, real_native), 'replay_vs_native_real': auroc(replay, real_native),
         'edit_tpr@0.2': float(np.mean(np.array(e) >= 0.2)), 'replay_tpr@0.2': float(np.mean(np.array(replay) >= 0.2)),
         'real_fpr@0.2': float(np.mean(np.array(real_native + real_matched) >= 0.2)), 'replay_vs_replay_real': auroc(replay, rr), 'replay_real_fpr@0.2': float(np.mean(np.array(rr) >= 0.2)), 'n': [len(e), len(real_native), len(real_matched), len(replay), len(rr)]}
    model.train(); return r
model = timm.create_model('vit_small_patch16_384', pretrained=False, num_classes=1, drop_path_rate=0.1)
model.load_state_dict({k.removeprefix('vit.'): v for k, v in load_file(hf_hub_download('OwensLab/commfor-model-384', 'model.safetensors')).items()})
model.to(dev)
ds = Train(items); dl = DataLoader(ds, BS, shuffle=True, num_workers=NW, drop_last=True, persistent_workers=NW > 0, pin_memory=dev == 'cuda')
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
steps = EPOCHS * len(dl); warm = max(1, steps // 20)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))
pos = sum(it['label'] for it in ds.items); print(f'{dev}: {len(ds)} crops per epoch ({pos} generated), {len(dl)} steps', flush=True)
log = [{'epoch': 0, **evaluate(model)}]; print(log[-1], flush=True)
for ep in range(1, EPOCHS + 1):
    t0 = time.time(); tot = 0
    for i, (x, y) in enumerate(dl):
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == 'cuda'):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x).float(), y)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); tot += loss.item()
    log.append({'epoch': ep, 'loss': tot / len(dl), 'sec': round(time.time() - t0), 'crops_per_s': round(len(dl) * BS / (time.time() - t0)), **evaluate(model)})
    print(log[-1], flush=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, f'{OUT}/ep{ep}.pt'); json.dump(log, open(f'{OUT}/log.json', 'w'))
