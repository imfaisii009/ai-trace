// Robustness test: does the model still see an AI image after ordinary camera-style post-processing?
// Applies each operation on its own and then all of them in a chain, scores every variant with the
// page's model, tiles and line, and writes one HTML report with the heatmaps before and after.
//
//   cd scripts && npm install
//   node robustness.mjs ../public/models/commfor-384-grok-ft-w8.onnx <out-dir> <image> [image...]
//
// The point is to find what the model is not robust to, so it can go into the training augmentation
// (docs/finetune.md). The report holds small previews only; keep <out-dir> outside the repository.
import ort from 'onnxruntime-node';
import sharp from 'sharp';
import { mkdirSync, writeFileSync } from 'node:fs';
import { basename, join } from 'node:path';

const [model, outDir, ...files] = process.argv.slice(2);
if (!model || !outDir || !files.length) {
  console.error('usage: node robustness.mjs <model.onnx> <out-dir> <image> [image...]');
  process.exit(1);
}
const sess = await ort.InferenceSession.create(model);
const N = 384;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const MAX_SIDE = 3072;
const LINE = 0.6; // same as MODEL.threshold in public/classify.js
const T = 24; // patch tokens per tile side
const CELL = 8; // heatmap cell in image pixels, as on the page
const PREVIEW = 360; // report preview width

// Seeded noise, so a run can be repeated exactly.
const rng = (a) => () => {
  a = (a + 0x6d2b79f5) | 0;
  let t = Math.imul(a ^ (a >>> 15), 1 | a);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};
const gauss = (r) => Math.sqrt(-2 * Math.log(r() || 1e-12)) * Math.cos(2 * Math.PI * r());

const raw = async (s) => {
  const { data, info } = await s.raw().toBuffer({ resolveWithObject: true });
  return { d: data, w: info.width, h: info.height };
};
const fromRaw = ({ d, w, h }) => sharp(d, { raw: { width: w, height: h, channels: 3 } });
const perPixel = ({ d, w, h }, f) => {
  const o = new Uint8ClampedArray(d.length);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) f(o, (y * w + x) * 3, x, y);
  return { d: o, w, h };
};

// Mild settings, of the size a camera pipeline or a photo app adds.
const OPS = {
  grain: (im) => {
    const r = rng(1);
    return perPixel(im, (o, p) => {
      const l = 4 * gauss(r); // luminance noise, sigma 4 of 255
      for (let c = 0; c < 3; c++) o[p + c] = im.d[p + c] + l + 2 * gauss(r); // plus chroma noise, sigma 2
    });
  },
  contrast: async (im) => raw(fromRaw(im).sharpen({ sigma: 3, m1: 0.3, m2: 0.6 })), // wide, gentle unsharp mask
  white: (im) => perPixel(im, (o, p) => { // warm white-balance drift
    o[p] = im.d[p] * 1.03;
    o[p + 1] = im.d[p + 1];
    o[p + 2] = im.d[p + 2] * 0.97;
  }),
  aberration: (im) => { // lateral chromatic aberration: red scaled up 0.15% about the centre, blue down
    const { d, w, h } = im;
    const cx = (w - 1) / 2;
    const cy = (h - 1) / 2;
    const sample = (x, y, c) => {
      const x0 = Math.max(0, Math.min(w - 2, Math.floor(x)));
      const y0 = Math.max(0, Math.min(h - 2, Math.floor(y)));
      const fx = Math.min(1, Math.max(0, x - x0));
      const fy = Math.min(1, Math.max(0, y - y0));
      const at = (xx, yy) => d[(yy * w + xx) * 3 + c];
      return (at(x0, y0) * (1 - fx) + at(x0 + 1, y0) * fx) * (1 - fy) + (at(x0, y0 + 1) * (1 - fx) + at(x0 + 1, y0 + 1) * fx) * fy;
    };
    return perPixel(im, (o, p, x, y) => {
      o[p] = sample(cx + (x - cx) / 1.0015, cy + (y - cy) / 1.0015, 0);
      o[p + 1] = d[p + 1];
      o[p + 2] = sample(cx + (x - cx) / 0.9985, cy + (y - cy) / 0.9985, 2);
    });
  },
  vignette: (im) => {
    const { w, h } = im;
    const r2 = (w / 2) ** 2 + (h / 2) ** 2;
    return perPixel(im, (o, p, x, y) => {
      const k = 1 - 0.18 * (((x - w / 2) ** 2 + (y - h / 2) ** 2) / r2);
      for (let c = 0; c < 3; c++) o[p + c] = im.d[p + c] * k;
    });
  },
  jpeg: async (im) => raw(sharp(await fromRaw(im).jpeg({ quality: 85 }).toBuffer())),
};
const CHAIN = ['grain', 'contrast', 'white', 'aberration', 'vignette', 'jpeg']; // the order in CLAUDE.md

function starts(len, step = N) {
  const s = [];
  for (let p = 0; p + N <= len; p += step) s.push(p);
  if (!s.length) s.push(0);
  else if (s.at(-1) !== len - N) s.push(len - N);
  return s;
}

// Heatmap, as in public/classify.js: each tile's token map divided by its maximum, times the tile
// score, sampled at each cell centre and blended across overlapping tiles with a tent weight.
const tent = (d) => Math.max(1e-3, 1 - Math.abs((2 * d) / N - 1));
function addTile(heat, x0, y0, tok, p) {
  let top = 0;
  for (const v of tok) top = Math.max(top, v);
  const k = top > 0 ? p / top : 0;
  const at = (r, c) => tok[Math.min(T - 1, Math.max(0, r)) * T + Math.min(T - 1, Math.max(0, c))];
  for (let j = Math.ceil(y0 / CELL); j < Math.min(heat.h, (y0 + N) / CELL); j++) {
    const dy = (j + 0.5) * CELL - y0;
    const v = dy / (N / T) - 0.5;
    const r = Math.floor(v);
    const fy = v - r;
    for (let i = Math.ceil(x0 / CELL); i < Math.min(heat.w, (x0 + N) / CELL); i++) {
      const dx = (i + 0.5) * CELL - x0;
      const u = dx / (N / T) - 0.5;
      const c = Math.floor(u);
      const fx = u - c;
      const s = (at(r, c) * (1 - fx) + at(r, c + 1) * fx) * (1 - fy) + (at(r + 1, c) * (1 - fx) + at(r + 1, c + 1) * fx) * fy;
      const wt = tent(dx) * tent(dy);
      heat.sum[j * heat.w + i] += s * k * wt;
      heat.n[j * heat.w + i] += wt;
    }
  }
}

// Score from the 384-pixel grid; heatmap from tiles every `step` pixels (192 on the page).
async function score({ d, w: W, h: H }, step) {
  const gy = new Set(starts(H));
  const gx = new Set(starts(W));
  const heat = { w: Math.ceil(W / CELL), h: Math.ceil(H / CELL) };
  heat.sum = new Float32Array(heat.w * heat.h);
  heat.n = new Float32Array(heat.w * heat.h);
  const grid = [];
  const x = new Float32Array(3 * N * N);
  for (const y0 of starts(H, step)) {
    for (const x0 of starts(W, step)) {
      for (let y = 0; y < N; y++) {
        const row = Math.min(H - 1, y0 + y) * W;
        for (let i = 0; i < N; i++) {
          const p = (row + Math.min(W - 1, x0 + i)) * 3;
          for (let c = 0; c < 3; c++) x[c * N * N + y * N + i] = (d[p + c] / 255 - MEAN[c]) / STD[c];
        }
      }
      const out = await sess.run({ pixel_values: new ort.Tensor('float32', x, [1, 3, N, N]) });
      const p = 1 / (1 + Math.exp(-out.logits.data[0]));
      if (gy.has(y0) && gx.has(x0)) grid.push(p);
      addTile(heat, x0, y0, out.token.data, p);
    }
  }
  for (let i = 0; i < heat.sum.length; i++) heat.sum[i] /= heat.n[i] || 1;
  let hot = 0;
  let sum = 0;
  for (const v of heat.sum) { sum += v; hot += v > 0.5; }
  return { score: grid.reduce((a, b) => a + b) / grid.length, heat, heatMean: sum / heat.sum.length, hot: hot / heat.sum.length };
}

// The page's jet scale at alpha 140, over a small preview of the image, as a data URI.
const jet = (v) => [3, 2, 1].map((c) => Math.round(255 * Math.min(1, Math.max(0, 1.5 - Math.abs(4 * v - c)))));
async function preview(im, heat) {
  const pw = PREVIEW;
  const ph = Math.round((im.h * pw) / im.w);
  const rgba = Buffer.alloc(heat.w * heat.h * 4);
  heat.sum.forEach((v, i) => rgba.set([...jet(v), 140], i * 4));
  const overlay = await sharp(rgba, { raw: { width: heat.w, height: heat.h, channels: 4 } })
    .resize(pw, ph, { fit: 'fill', kernel: 'cubic' }).png().toBuffer();
  const jpg = await fromRaw(im).resize(pw, ph).composite([{ input: overlay }]).jpeg({ quality: 80 }).toBuffer();
  return `data:image/jpeg;base64,${jpg.toString('base64')}`;
}

const pct = (v) => `${Math.round(v * 100)}%`;
const rows = [];
for (const f of files) {
  let img = sharp(f).rotate().removeAlpha().toColourspace('srgb');
  const meta = await sharp(f).metadata();
  const portrait = (meta.orientation ?? 1) >= 5;
  const [w, h] = portrait ? [meta.height, meta.width] : [meta.width, meta.height];
  if (Math.max(w, h) > MAX_SIDE) img = img.resize({ width: w >= h ? MAX_SIDE : undefined, height: h > w ? MAX_SIDE : undefined, kernel: 'cubic' });
  const orig = await raw(img);
  const t0 = Date.now();
  const base = await score(orig, N / 2);
  const single = {};
  for (const [name, op] of Object.entries(OPS)) single[name] = (await score(await op(orig), N)).score;
  let chained = orig;
  for (const name of CHAIN) chained = await OPS[name](chained);
  const after = await score(chained, N / 2);
  rows.push({ name: basename(f), w: orig.w, h: orig.h, base, single, after, prev: [await preview(orig, base.heat), await preview(chained, after.heat)] });
  console.log(`${basename(f).slice(0, 40).padEnd(40)} ${orig.w}x${orig.h} orig=${base.score.toFixed(3)} ${Object.entries(single).map(([k, v]) => `${k}=${v.toFixed(3)}`).join(' ')} chain=${after.score.toFixed(3)} hot ${pct(base.hot)}->${pct(after.hot)} (${((Date.now() - t0) / 1000).toFixed(0)} s)`);
}

const avg = (f) => rows.reduce((s, r) => s + f(r), 0) / rows.length;
const over = (f) => rows.filter((r) => f(r) >= LINE).length;
const opNames = Object.keys(OPS);
const cell = (v) => `<td class="${v >= LINE ? 'ai' : 'real'}">${pct(v)}</td>`;
const html = `<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Robustness test</title>
<style>
:root{--bg:#f6f7f8;--fg:#1b2226;--muted:#5b676d;--line:#d3d9dc;--ai:#b3261e;--real:#1f6f43}
@media (prefers-color-scheme:dark){:root{--bg:#111517;--fg:#e6ebed;--muted:#9ba7ad;--line:#2b3438;--ai:#ff8a80;--real:#7bd88f}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0;padding:24px 16px}
main{max-width:1100px;margin:0 auto}h1{margin:0 0 4px}p{color:var(--muted);max-width:75ch}
.scroll{overflow-x:auto}table{border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:13px}
th,td{padding:4px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}
td.ai{color:var(--ai)}td.real{color:var(--real)}
.pair{display:flex;flex-wrap:wrap;gap:12px;margin:20px 0;border-top:1px solid var(--line);padding-top:12px}
figure{margin:0;max-width:${PREVIEW}px}figure img{width:100%;height:auto;display:block}figcaption{font-size:13px;color:var(--muted)}
.name{width:100%;font-weight:600;font-size:14px}
</style><main>
<h1>Robustness test</h1>
<p>${rows.length} images, scored with ${basename(model)} at the page's settings (384-pixel grid, line ${pct(LINE)}, heatmap tiles every 192 pixels), ${new Date().toISOString().slice(0, 10)}. Each operation is applied alone, then all six in a chain: ${CHAIN.join(', ')}. Red is at or over the line.</p>
<p>Mean score: original ${pct(avg((r) => r.base.score))}, after the chain ${pct(avg((r) => r.after.score))}. Over the line: ${over((r) => r.base.score)} of ${rows.length} before, ${over((r) => r.after.score)} after. Share of the heatmap over 0.5: ${pct(avg((r) => r.base.hot))} before, ${pct(avg((r) => r.after.hot))} after.</p>
<div class="scroll"><table><tr><th>Image</th><th>Original</th>${opNames.map((n) => `<th>${n}</th>`).join('')}<th>Chain</th><th>Map &gt; 0.5 before</th><th>after</th></tr>
${rows.map((r) => `<tr><td>${r.name.slice(0, 32)}</td>${cell(r.base.score)}${opNames.map((n) => cell(r.single[n])).join('')}${cell(r.after.score)}<td>${pct(r.base.hot)}</td><td>${pct(r.after.hot)}</td></tr>`).join('\n')}
<tr><th>Mean</th>${cell(avg((r) => r.base.score))}${opNames.map((n) => cell(avg((r) => r.single[n]))).join('')}${cell(avg((r) => r.after.score))}<th>${pct(avg((r) => r.base.hot))}</th><th>${pct(avg((r) => r.after.hot))}</th></tr>
</table></div>
${rows.map((r) => `<div class="pair"><div class="name">${r.name} · ${r.w} × ${r.h}</div>
<figure><img src="${r.prev[0]}" alt="Heatmap on the original"><figcaption>Original: ${pct(r.base.score)}, map mean ${r.base.heatMean.toFixed(2)}</figcaption></figure>
<figure><img src="${r.prev[1]}" alt="Heatmap after the chain"><figcaption>After the chain: ${pct(r.after.score)}, map mean ${r.after.heatMean.toFixed(2)}</figcaption></figure></div>`).join('\n')}
</main>`;
mkdirSync(outDir, { recursive: true });
writeFileSync(join(outDir, 'robustness.html'), html);
console.log(`\nmean orig=${avg((r) => r.base.score).toFixed(3)} chain=${avg((r) => r.after.score).toFixed(3)}; over line ${over((r) => r.base.score)}/${rows.length} -> ${over((r) => r.after.score)}/${rows.length}\nreport: ${join(outDir, 'robustness.html')}`);
