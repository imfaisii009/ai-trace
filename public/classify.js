// The pixel check. Community Forensics (Park and Owens, CVPR 2025, MIT licence) is a ViT-S/16 that
// looks at a 384 x 384 crop and says how likely it is to come from a generator. It was trained on
// images from 4,803 generators. We fine-tuned it on phone photos paired with AI edits of the same
// photos (docs/finetune.md), which is what lets it see a whole-photo edit, not just a fully
// generated image. Its weights are stored as 8-bit integers (23 MB) and it runs with ONNX Runtime
// Web, so the image never leaves the browser.
//
// The model scores 384-pixel tiles at the image's own resolution, because an edit that keeps the
// layout of a real photo only shows in fine texture, which shrinking the image to 384 pixels wipes
// out. The image score is the mean of the tile scores.
//
// The heatmap: the exported model also returns, for each tile, the classifier head applied to each
// of its 24 x 24 patch tokens (16 pixels each). Each tile's token map is divided by its own maximum
// and multiplied by the tile score, so a tile the model calls camera-made stays cold however its
// tokens rank, then stitched into one grid where overlapping tiles blend.

const ORT = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
export const MODEL = {
  url: new URL('./models/commfor-384-grok-ft-w8.onnx', import.meta.url).href,
  size: '23 MB',
  threshold: 0.6, // docs/finetune.md: no held-out phone photo and 1.3% of other real photos reach it
};
const N = 384;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const MAX_SIDE = 3072; // ponytail: larger images are scaled down, which caps a run at about 165 tiles
const T = 24; // patch tokens per tile side, 16 pixels each
const CELL = 8; // heatmap grid cell in image pixels

let session = null;
let queue = Promise.resolve();
export const loaded = () => !!session;

async function load() {
  const ort = await import(`${ORT}ort.wasm.min.mjs`);
  ort.env.wasm.wasmPaths = ORT;
  ort.env.wasm.proxy = true; // runs the model in a worker, so the page stays responsive meanwhile
  return { ort, run: await ort.InferenceSession.create(MODEL.url) };
}

// Tile origins along one side: every `step` pixels, plus one flush with the far edge.
function starts(len, step = N) {
  const s = [];
  for (let p = 0; p + N <= len; p += step) s.push(p);
  if (!s.length) s.push(0);
  else if (s.at(-1) !== len - N) s.push(len - N);
  return s;
}

// Tent weight of a point `d` pixels into a tile: 1 at the centre, falling to 0 at the edges. Overlapping
// tiles blend into each other instead of leaving a seam where one ends.
const tent = (d) => Math.max(1e-3, 1 - Math.abs((2 * d) / N - 1));

// Adds one tile's token map to the grid: scaled to the tile score, sampled bilinearly at each cell
// centre inside the tile, weighted by its distance from the tile edge.
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
      const w = tent(dx) * tent(dy);
      heat.sum[j * heat.w + i] += s * k * w;
      heat.n[j * heat.w + i] += w;
    }
  }
}

/** Decodes an image as the model sees it: EXIF rotation applied, scaled down to MAX_SIDE. */
export async function decode(blob) {
  const bmp = await createImageBitmap(blob);
  const k = Math.min(1, MAX_SIDE / Math.max(bmp.width, bmp.height));
  const W = Math.round(bmp.width * k);
  const H = Math.round(bmp.height * k);
  const ctx = Object.assign(document.createElement('canvas'), { width: W, height: H }).getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(bmp, 0, 0, W, H);
  bmp.close();
  return { px: ctx.getImageData(0, 0, W, H).data, width: W, height: H, scaled: k < 1 };
}

// Tiles overlap by half (a step of 192 pixels), which halves the seams between tiles in the
// heatmap: on pasted generated patches its AUROC rose from 0.62 to 0.74. The score still comes from
// the tiles on the 384-pixel grid, a subset of these, which is what the tests were run on.
async function score({ px, width: W, height: H, scaled }, progress) {
  if (!session) session = load().catch((e) => { session = null; throw e; });
  const { ort, run } = await session;
  const ys = starts(H, N / 2);
  const xs = starts(W, N / 2);
  const gy = new Set(starts(H));
  const gx = new Set(starts(W));
  const tiles = [];
  const heat = { w: Math.ceil(W / CELL), h: Math.ceil(H / CELL) };
  heat.sum = new Float32Array(heat.w * heat.h);
  heat.n = new Float32Array(heat.w * heat.h);
  const t0 = performance.now();
  for (const y0 of ys) {
    for (const x0 of xs) {
      progress(tiles.length + 1, ys.length * xs.length);
      const x = new Float32Array(3 * N * N); // a new buffer each time: the worker takes ownership of it
      for (let y = 0; y < N; y++) {
        const row = Math.min(H - 1, y0 + y) * W;
        for (let i = 0; i < N; i++) {
          const p = (row + Math.min(W - 1, x0 + i)) * 4;
          const q = y * N + i;
          for (let c = 0; c < 3; c++) x[c * N * N + q] = (px[p + c] / 255 - MEAN[c]) / STD[c];
        }
      }
      const out = await run.run({ pixel_values: new ort.Tensor('float32', x, [1, 3, N, N]) });
      const p = 1 / (1 + Math.exp(-out.logits.data[0]));
      tiles.push({ score: p, grid: gy.has(y0) && gx.has(x0) });
      addTile(heat, x0, y0, out.token.data, p);
    }
  }
  for (let i = 0; i < heat.sum.length; i++) heat.sum[i] /= heat.n[i] || 1;
  const grid = tiles.filter((t) => t.grid);
  return {
    score: grid.reduce((s, t) => s + t.score, 0) / grid.length,
    tiles,
    heat: { w: heat.w, h: heat.h, v: heat.sum },
    width: W,
    height: H,
    scaled,
    ms: performance.now() - t0,
  };
}

/** Runs one image at a time: an ONNX session cannot run two calls at once. */
export function classify(img, progress = () => {}) {
  const job = queue.then(() => score(img, progress));
  queue = job.catch(() => {});
  return job;
}
