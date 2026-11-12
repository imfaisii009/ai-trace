// Scores images with the same model, tiles and line as the page, outside the browser, so a
// change to the model or the method can be measured on a folder of images.
//
//   cd scripts && npm install
//   node eval.mjs ../public/models/commfor-384-grok-ft-w8.onnx path/to/images/*.jpg
//
// Prints one line per image: the verdict, size, tile count, mean and highest tile score.
import ort from 'onnxruntime-node';
import sharp from 'sharp';

const [model, ...files] = process.argv.slice(2);
if (!model || !files.length) {
  console.error('usage: node eval.mjs <model.onnx> <image> [image...]');
  process.exit(1);
}
const sess = await ort.InferenceSession.create(model);
const N = 384;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const MAX_SIDE = 3072;
const LINE = 0.6; // same as MODEL.threshold in public/classify.js

// The tiles classify.js scores from: one every 384 pixels, plus one flush with the far edge. (The page
// also runs the tiles in between, every 192 pixels, but only for the heatmap.)
const starts = (len) => {
  const s = [];
  for (let p = 0; p + N <= len; p += N) s.push(p);
  if (!s.length) s.push(0);
  else if (s.at(-1) !== len - N) s.push(len - N);
  return s;
};

for (const f of files) {
  let img = sharp(f).rotate().removeAlpha();
  const meta = await img.metadata();
  const portrait = (meta.orientation ?? 1) >= 5;
  const [w, h] = portrait ? [meta.height, meta.width] : [meta.width, meta.height];
  if (Math.max(w, h) > MAX_SIDE) img = img.resize({ width: w >= h ? MAX_SIDE : undefined, height: h > w ? MAX_SIDE : undefined, kernel: 'cubic' });
  const { data, info } = await img.raw().toBuffer({ resolveWithObject: true });
  const W = info.width;
  const H = info.height;
  const scores = [];
  const t0 = Date.now();
  const x = new Float32Array(3 * N * N);
  for (const y0 of starts(H)) {
    for (const x0 of starts(W)) {
      for (let y = 0; y < N; y++) {
        for (let i = 0; i < N; i++) {
          const p = (Math.min(H - 1, y0 + y) * W + Math.min(W - 1, x0 + i)) * 3;
          for (let c = 0; c < 3; c++) x[c * N * N + y * N + i] = (data[p + c] / 255 - MEAN[c]) / STD[c];
        }
      }
      const out = await sess.run({ pixel_values: new ort.Tensor('float32', x, [1, 3, N, N]) });
      scores.push(1 / (1 + Math.exp(-out.logits.data[0])));
    }
  }
  const mean = scores.reduce((a, b) => a + b) / scores.length;
  const name = f.split('/').slice(-2).join('/').slice(0, 40).padEnd(40);
  console.log(`${mean >= LINE ? 'AI  ' : 'real'} ${name} ${W}x${H} tiles=${scores.length} mean=${mean.toFixed(3)} max=${Math.max(...scores).toFixed(3)} (${Date.now() - t0} ms)`);
}
