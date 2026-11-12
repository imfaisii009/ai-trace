# AI Trace

Find the traces AI tools leave in text and images, see the evidence behind each one, and compare an image with its metadata-free copy under the same AI-image model.

Everything runs in the browser. Nothing you paste or open is uploaded: the page is static files, and the AI-image model runs on your own CPU with ONNX Runtime Web.

**Try it:** https://ai-trace.imfaisii009.workers.dev

## What it does

### Text

Paste text or open a `.txt`/`.md` file. Every finding is marked in place, grouped by rule with line and column, and can be left in or removed from a clean copy.

| Category | Examples | Cleaned by default |
|---|---|---|
| Hidden characters | Zero-width space and joiners, word joiner, BOM, soft hyphen, direction controls, private-use characters, Unicode tag characters, variation selectors | Yes |
| Chat-app artifacts | ChatGPT citation markers (`citeturn0search1`, `【4:0†source】`, `:contentReference[oaicite:0]{index=0}`), `utm_source=chatgpt.com` and other AI referral parameters | Yes |
| Unusual spaces | Non-breaking, narrow, thin, em and ideographic spaces; Unicode line separators | Yes |
| Look-alike letters | Cyrillic or Greek letters inside Latin words | Yes |
| Typography | Curly quotes, em and en dashes, the ellipsis character | No |

Messages hidden in Unicode tag characters or variation selectors ("emoji smuggling") are decoded and shown, which also catches hidden prompt injections. Normal uses are left alone: joiners inside emoji sequences and in Arabic and Indic scripts, emoji presentation selectors, keycaps and flags.

### Images (JPEG, PNG, WebP)

The image report has five parts.

1. **AI signals in the metadata.** C2PA content credentials (generator, signer, actions, soft bindings), the IPTC label `trainedAlgorithmicMedia`, and AI tool names (OpenAI, Google, xAI, Adobe Firefly, Midjourney, Stable Diffusion, Flux and others), each quoted with the block and byte offset it came from.
2. **Pixel check.** Format hints a pixel classifier may also pick up: image size on a generator's grid, the JPEG encoder and quality read from the quantisation tables, PNG or WebP output, and whether a camera is named in EXIF. Then the AI-image model (below).
3. **Metadata blocks.** Every block with its byte range, what it says, how it was found, a hex dump, and for C2PA the JUMBF box tree.
4. **Watermarks this page cannot read.** SynthID, Adobe TrustMark, Meta's Video Seal and Stable Signature: what each is and whether the metadata hints at it.
5. **Original and clean copy.** The cleaner copies the image data byte for byte and drops every metadata block (colour profiles and blocks a decoder needs are kept, and a rotated JPEG keeps its rotation). The clean copy is scanned again to show what is left, then the model scores both copies and draws a heatmap on each, side by side.

The comparison is there to show what cleaning does and does not change. It removes what the file says about itself. It does not touch the pixels, so a pixel detector, this one included, gives the clean copy the same score.

## How the AI-image model and the heatmap work

The model is [Community Forensics](https://github.com/JeongsooP/Community-Forensics) (Jeongsoo Park and Andrew Owens, CVPR 2025): a ViT-S/16 vision transformer trained on images from 4,803 generators, which outputs one logit for "generated". We fine-tuned it on phone photos paired with AI edits of the same photos, so it also sees a whole-photo edit, not only a fully generated image ([how, and the numbers](docs/finetune.md)). It is exported to ONNX with a second output for the heatmap, its weights stored as 8-bit integers (23 MB, `public/models/commfor-384-grok-ft-w8.onnx`). `scripts/export-model.py` builds it from a checkpoint, or from the published weights.

`public/classify.js` does the work, in a worker so the page stays responsive:

1. **Decode.** The image is decoded by the browser (`createImageBitmap`, which applies the EXIF rotation) and drawn to a canvas. Images over 3,072 pixels on the long side are scaled down first.
2. **Compare.** The original and the clean copy are decoded the same way and compared pixel by pixel. If they match, which is the usual case, the model runs once and both get its result, since identical input gives identical output. If not, both are scored and the page reports the largest change in one tile.
3. **Tile.** The model sees 384 × 384 pixels at a time. Shrinking a whole photo to 384 pixels wipes out the fine texture an AI edit leaves behind, so the image is cut into tiles at its own resolution, one every 192 pixels along each side plus one flush with the far edge. An image smaller than 384 pixels on a side is padded by repeating its edge pixels.
4. **Score.** Each tile is normalised with the ImageNet mean and standard deviation and run through the model. A sigmoid turns the logit into a score from 0 to 1. The image score is the mean over the tiles on the 384-pixel grid (every other tile, plus the flush ones), which is the setting the tests below were run on. At 60% or more the page says the image looks generated. The line was set from the tests in [docs/finetune.md](docs/finetune.md).
5. **Map.** Inside a tile the ViT works on 24 × 24 patches of 16 pixels, and its verdict is read from one summary token that gathers them. The exported model also applies the same classifier head to every patch token (`token`, a 24 × 24 grid of sigmoids), so each patch gets a score from the model's own last layer. Each tile's grid is divided by its maximum and multiplied by the tile's score, so a tile the model calls camera-made stays cold however its patches rank. The grids are sampled bilinearly onto an 8-pixel grid over the image and overlapping tiles are blended with tent weights (1 at a tile's centre, 0 at its edge), so tile edges leave no seams.
6. **Draw.** `paintHeat` in `public/app.js` colours each cell on the jet scale (blue 0, red 1) at a fixed opacity, and the browser smooths it as it scales the canvas over the image. The scale is absolute: a photo the model calls real stays blue rather than being stretched to show its least-real corner in red.

### How well it works

Tested on 11 November 2026 with the page's settings, on images that share nothing with training: it called 102 of 102 Grok edits of phone photos generated (58% before fine-tuning), 20 of 20 Grok text-to-image photos, 101 of 101 other generated images, and 6 of 254 real images (0 of 80 phone photos, 1 of 150 real images from the Community Forensics test set, 5 of 24 lossless film scans). The full table is in [docs/finetune.md](docs/finetune.md). Edits from tools other than Grok were not in training, and film scans and screenshots are the weak spot. Treat a score as a pointer, not proof. The full table, and how online detectors such as DeepAI and TruthScan compare, are on the page under How detection works.

The heatmap was chosen by measurement, not by eye, on the published weights before fine-tuning. Three tests, all in the [heatmap evaluation](docs/heatmap-evaluation.md):

- **Pasted patches.** Squares of 448 or 512 pixels cut from generated images and pasted into 24 real photos, saved as PNG and as JPEG, 48 composites with the right answer known for every pixel. Measured: how well the map ranks pasted pixels above the rest (AUROC, 0.5 is chance).
- **Deletion.** Blur the 20% of a generated image the map rates highest and score it again. A faithful map makes the score drop more than blurring random 32-pixel blocks does.
- **Real photos.** How much of a real photo the map paints hot.

Six methods were compared (the tile score itself, the patch-token head, gradient × input, Grad-CAM, saliency and attention rollout). The patch-token head, scaled by the tile score, found pasted patches as well as the tile scores did (AUROC 0.75 against 0.75) at 16-pixel instead of 384-pixel resolution, and better than gradient × input (0.70) or Grad-CAM (0.64). Gradient × input was more faithful on the deletion test, but needs a backward pass at about three times the cost per tile. The patch-token map comes out of the same forward pass as the score. Tiles every 192 pixels instead of 384 raised its AUROC on the pasted patches from 0.62 to 0.74 and halved the false heat on real photos.

The limit is the model, not the map: patches under about 320 pixels were not detected at all (the image scored about 3%), and even 448 to 512 pixel patches were detected in only about a quarter of cases. Community Forensics recognises whole generated or regenerated images far better than a pasted region. A map cannot point at evidence the model did not find.

## Run it

It is a static site with no build step.

```bash
npx wrangler dev
```

Open the URL it prints. `public/_headers` makes the page cross-origin isolated, which lets ONNX Runtime Web use several threads (about three times faster). Any static file server works too, for example `python3 -m http.server -d public`, but without those two headers the model runs on one thread.

Deploy to Cloudflare Workers (static assets only, no Worker code):

```bash
npx wrangler deploy
```

Any static host works if it can send the two headers in `public/_headers` and serve a 23 MB file.

## Project layout

```
public/
  index.html        page, styles, and the How detection works write-up
  app.js            UI: text and image reports, heatmaps, comparison
  scan-text.js      text rules, findings and the clean copy
  scan-file.js      JPEG, PNG and WebP block walker, cleaner, format hints
  c2pa.js           C2PA reader: JUMBF boxes, CBOR, COSE signature fields
  classify.js       the AI-image model: tiles, scores, the heatmap grid
  models/           the fine-tuned Community Forensics model, with its patch-score output
  _headers          cross-origin isolation headers
scripts/
  export-model.py   build the ONNX model from a checkpoint or the published weights
  eval.mjs          score a folder of images with the same model and settings
  finetune/         collect photos, make edit jobs, train, score (docs/finetune.md)
docs/
  finetune.md       how the model was fine-tuned, with the test results
  heatmap-evaluation.md  how the heatmap method was chosen
```

ONNX Runtime Web 1.20.1 and the Google Fonts stylesheet load from CDNs. Nothing else does.

## Measure a change

`scripts/eval.mjs` uses the same tiles, normalisation and line as the page, in Node:

```bash
cd scripts && npm install
node eval.mjs ../public/models/commfor-384-grok-ft-w8.onnx ~/images/real/*.jpg ~/images/ai/*.png
```

A pull request that changes the model, the tiles or the line should include before and after counts on real and generated images.

## Limits

- Missing metadata proves nothing: most social sites and messaging apps strip it on upload, and many generators write none.
- A tool name in metadata is a lead, not proof: a Google camera writes "Google" too.
- C2PA signatures and hashes are read, not verified.
- Pixel watermarks (SynthID and others) need their owners' decoders; the page only reports what the metadata says about them.
- Only JPEG, PNG and WebP are parsed block by block. Other files get a string scan.

## Contributing

See [ROADMAP.md](ROADMAP.md) for open work. Keep everything in the browser: no uploads and no servers.

## Licence

MIT, see [LICENSE](LICENSE). The bundled model weights and ONNX Runtime Web are MIT-licensed by their authors, and the fine-tuning data is credited in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
