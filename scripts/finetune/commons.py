# List public phone photos on Wikimedia Commons from the largest "Taken with <phone>" categories.
import urllib.error, json, sys, random, time, urllib.request, urllib.parse
OUT = sys.argv[1]; TARGET = int(sys.argv[2]) if len(sys.argv) > 2 else 1300; PER_CAT = 30
UA = {'User-Agent': 'ai-trace-dataset-builder/0.1 (research; detector training)'}
def api(**p):
    p.update(format='json', formatversion=2)
    for i in range(8):
        try:
            with urllib.request.urlopen(urllib.request.Request('https://commons.wikimedia.org/w/api.php?' + urllib.parse.urlencode(p), headers=UA), timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            time.sleep(int(e.headers.get('Retry-After') or 0) + 5 * (i + 1)); err = e
        except Exception as e:
            time.sleep(5 * (i + 1)); err = e
    raise err
cats = []
import os
for q in os.environ.get('BRANDS', 'Google Pixel,Samsung Galaxy,Xiaomi,OnePlus,Huawei,Motorola,Oppo,Sony Xperia,Nokia,Honor,Vivo,Realme,LG').split(','):
    r = api(action='query', list='allcategories', acprefix='Taken with ' + q, aclimit=500, acprop='size', acmin=40)
    cats += [(c['files'], 'Category:' + c['category'], q) for c in r.get('query', {}).get('allcategories', []) if c.get('files', 0) >= 40]
    time.sleep(1)
# round-robin over brands, biggest categories first, so no single phone dominates
by = {}
for f, c, b in sorted(cats, reverse=True): by.setdefault(b, []).append(c)
order = [c for i in range(max(map(len, by.values()))) for b in by.values() if i < len(b) for c in [b[i]]]
print(len(order), 'categories with 40+ files;', {b: len(v) for b, v in by.items()}, file=sys.stderr, flush=True)
out, seen = [], set()
for cat in order:
    if len(out) >= TARGET: break
    n, cont = 0, {}
    for _ in range(4):
        time.sleep(1.2)
        r = api(action='query', generator='categorymembers', gcmtitle=cat, gcmtype='file', gcmlimit=50, gcmsort='timestamp', gcmdir='desc',
                prop='imageinfo', iiprop='url|size|mime|sha1', iiurlwidth=1280, **cont)
        for p in (r.get('query', {}).get('pages') or []):
            ii = (p.get('imageinfo') or [{}])[0]
            if ii.get('mime') != 'image/jpeg' or min(ii.get('width', 0), ii.get('height', 0)) < 1500 or ii['sha1'] in seen or n >= PER_CAT: continue
            seen.add(ii['sha1']); n += 1
            out.append({'title': p['title'], 'cat': cat, 'url': ii['url'], 'thumb': ii.get('thumburl'), 'w': ii['width'], 'h': ii['height'], 'sha1': ii['sha1']})
        cont = r.get('continue') or {}
        if n >= PER_CAT or 'gcmcontinue' not in cont: break
        cont = {'gcmcontinue': cont['gcmcontinue']}
    print(cat, n, len(out), file=sys.stderr, flush=True)
random.seed(1); random.shuffle(out); json.dump(out, open(OUT, 'w'), indent=0)
print(len(out), 'photos from', len({o['cat'] for o in out}), 'categories', file=sys.stderr)
