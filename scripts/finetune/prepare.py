# Runs on the GPU box. Builds /data from: commons.json (public phone photos), edits/ + edits.json (Grok edits of
# some of them), and two Community Forensics shards (CC-BY-4.0) for replay, so the model keeps what it knew.
# Output: manifest.json listing every training/validation file with its label and split.
import json, os, sys, io, base64, random, time, urllib.request, urllib.error, concurrent.futures as cf
import pandas as pd
from PIL import Image, ImageOps
from huggingface_hub import hf_hub_download
D = sys.argv[1]; N = 384; random.seed(0)
UA = {'User-Agent': 'ai-trace-dataset-builder/0.1 (research; detector training)'}
for d in ('orig', 'real', 'replay'): os.makedirs(f'{D}/{d}', exist_ok=True)
os.system(f'rm -f {D}/replay/*')  # rebuilt each run
src = json.load(open(f'{D}/commons.json'))
edits = json.load(open(f'{D}/edits.json'))  # {sha1: [file names in D/edits]}
val = lambda sha: int(sha[:8], 16) % 100 < 15
def fetch(o):
    p = f"{D}/orig/{o['sha1']}.jpg"
    if os.path.exists(p): return o, p
    if os.environ.get('NO_FETCH'): return o, None  # use only photos already on disk
    for i in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(o['url'], headers=UA), timeout=180) as r:
                open(p, 'wb').write(r.read()); return o, p
        except urllib.error.HTTPError as e: time.sleep(int(e.headers.get('Retry-After') or 0) or 5 * (i + 1))
        except Exception: time.sleep(5 * (i + 1))
    return o, None
def cut(o, p):
    try: im = ImageOps.exif_transpose(Image.open(p)).convert('RGB')
    except Exception: return []
    W, H = im.size; sha = o['sha1']; out = []
    for k in range(10 if min(W, H) >= N else 0):  # panoramas thinner than a tile get no native crops
        x0, y0 = random.randint(0, W - N), random.randint(0, H - N)
        f = f'{D}/real/{sha}_n{k}.png'; im.crop((x0, y0, x0 + N, y0 + N)).save(f); out.append(('crop', f))
    s = 1600 / max(W, H); r = im.resize((round(W * s), round(H * s)), Image.LANCZOS) if s < 1 else im
    f = f'{D}/real/{sha}_r.png'; r.save(f); out.append(('master', f))
    return out
def raw(x):
    # image_data is base64 text in some shards and raw or base64 bytes in others
    if isinstance(x, (bytes, bytearray)):
        if x[:4] == b'\x89PNG' or x[:2] == b'\xff\xd8' or x[8:12] == b'WEBP': return bytes(x)
        x = x.decode()
    return base64.b64decode(x.strip('"'))
items = []
with cf.ThreadPoolExecutor(4) as ex: got = list(ex.map(fetch, src))
print('downloaded', sum(p is not None for _, p in got), 'of', len(src), flush=True)
with cf.ProcessPoolExecutor(os.cpu_count()) as ex:
    futs = {ex.submit(cut, o, p): o for o, p in got if p}
    for fu in cf.as_completed(futs):
        o = futs[fu]; split = 'val' if val(o['sha1']) else 'train'
        for kind, f in fu.result(): items.append({'path': f, 'label': 0, 'kind': kind, 'split': split, 'group': o['sha1'], 'orig': f"{D}/orig/{o['sha1']}.jpg"})
for sha, fs in edits.items():
    for f in fs: items.append({'path': f'{D}/edits/{f}', 'label': 1, 'kind': 'edit', 'split': 'val' if val(sha) else 'train', 'group': sha})
n = 0
for shard, take in (('data/manual/HFCF_Manual_0.parquet', 300), ('data/commercial/HFCF_Commercial_0.parquet', 1500)):
    df = pd.read_parquet(hf_hub_download('OwensLab/CommunityForensics', shard, repo_type='dataset'), columns=['image_data', 'label', 'nsfw_flag', 'model_name', 'format'])
    df = df[(df.label == 1) & (df.nsfw_flag.astype(str).str.lower() != 'true')].sample(frac=1, random_state=0).head(take)
    print(shard, len(df), 'generators:', df.model_name.nunique(), flush=True)
    for r in df.itertuples():
        b = raw(r.image_data)
        ext = 'png' if b[:4] == b'\x89PNG' else 'jpg' if b[:2] == b'\xff\xd8' else 'webp' if b[8:12] == b'WEBP' else 'bin'
        f = f'{D}/replay/{n}.{ext}'; open(f, 'wb').write(b)
        items.append({'path': f, 'label': 1, 'kind': 'replay', 'split': 'val' if n % 10 == 0 else 'train', 'group': f'r{n}', 'gen': r.model_name}); n += 1
# Eval shard: real photos (kept as real-side replay, so the model keeps its idea of "real") and modern generators.
df = pd.read_parquet(hf_hub_download('OwensLab/CommunityForensics', 'data/publicEval/HFCF_PublicEval_0.parquet', repo_type='dataset'), columns=['image_data', 'label', 'nsfw_flag', 'model_name'])
df = df[df.nsfw_flag.astype(str).str.lower() != 'true']
for lab, take, kind in ((0, 5000, 'replay_real'), (1, 600, 'replay')):
    d = df[df.label == lab].sample(frac=1, random_state=1).head(take)
    print('eval', kind, len(d), flush=True)
    for r in d.itertuples():
        b = raw(r.image_data)
        ext = 'png' if b[:4] == b'\x89PNG' else 'jpg' if b[:2] == b'\xff\xd8' else 'webp' if b[8:12] == b'WEBP' else 'bin'
        f = f'{D}/replay/{n}.{ext}'; open(f, 'wb').write(b)
        items.append({'path': f, 'label': lab, 'kind': kind, 'split': 'val' if n % 10 == 0 else 'train', 'group': f'r{n}', 'gen': r.model_name}); n += 1
del df
json.dump(items, open(f'{D}/manifest.json', 'w'))
from collections import Counter
print(Counter((i['split'], i['kind']) for i in items))
