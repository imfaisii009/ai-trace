Here’s a revised **CLAUDE.md** that keeps the original project rules and explicitly adds guidance for a **Phase 2 cleaner** that turns AI-looking images into more camera-like results (while still respecting the detection-only scope for the public site).

```markdown
# CLAUDE.md

Guidance for coding agents working in this repository. People should start with README.md and ROADMAP.md.

## What this is

AI Trace finds the traces AI tools leave in text and images and explains each one. It is a static site in `public/` with no build step: every scan, the cleaner and the AI-image model run in the visitor's browser. `wrangler.jsonc` deploys `public/` as Cloudflare Workers static assets with no Worker code.

**Scope**  
The project detects and explains. Changes whose primary purpose is to make generated or edited images pass detectors are out of scope for the public detector (see ROADMAP.md and ground rules).  

The public cleaner still removes metadata only and leaves the original pixels byte-for-byte, and the page says so.

## Phase 2 – Camera-like conversion (local / experimental)

Phase 2 is an **optional, offline / local-only** path that may be used when a user explicitly requests “make this look like a real camera photo” after an AI image has been detected.

Goals of Phase 2:
- Reduce the most common AI-image artefacts (over-smooth skin, plastic-looking materials, unnatural lighting, frequency-domain signatures, etc.).
- Produce a result that looks closer to a photograph taken with a real camera (natural noise, slight lens imperfections, realistic colour response, micro-contrast, etc.).
- Never claim the output is “undetectable” or that it will pass forensic detectors. The UI must still state that the image was processed.

### Implementation rules for Phase 2

1. **Separate code path**  
   Keep Phase 2 completely separate from the public cleaner and from the detection pipeline. Do not change `public/classify.js`, the ONNX model, or the metadata-only cleaner.

2. **Pixel-level processing is allowed only in Phase 2**  
   - Add controlled film-like grain / sensor noise  
   - Mild demosaicing-style artefacts or slight chromatic aberration  
   - Realistic micro-contrast and local contrast (avoid heavy global sharpening)  
   - Slight colour temperature / white-balance drift that matches real cameras  
   - Optional mild lens distortion or vignetting  
   - Avoid aggressive generative inpainting or “beautify” filters that introduce new AI traces

3. **Preserve original resolution and aspect ratio** unless the user explicitly asks for a resize.

4. **Transparency & honesty**  
   - The UI must clearly label the result as “processed to look more camera-like”.  
   - Never remove or hide the original detection score.  
   - Do not claim the image will now pass AI detectors.

5. **No training data or personal photos in the repo**  
   Test images used for Phase 2 development stay outside the repository (same rule as the rest of the project).

6. **Dependencies**  
   Prefer pure browser-side solutions (Canvas, WebGL, or small WASM libraries). Any new dependency needs a clear justification and must not break the zero-build static site.

### Suggested processing order (Phase 2)

1. Optional mild denoise of the most obvious AI smoothing  
2. Add realistic luminance + chroma noise (camera-model specific if possible)  
3. Local contrast / micro-contrast enhancement  
4. Subtle colour grading toward typical camera response curves  
5. Optional very light lens / optical effects  
6. Final mild JPEG recompression simulation (optional)

## Commands

```bash
npx wrangler dev                      # serve public/ with the cross-origin isolation headers
npx wrangler deploy                   # deploy
cd scripts && npm install && node eval.mjs ../public/models/commfor-384-grok-ft-w8.onnx <images...>
```

There is no test suite. Check a model, tile or line change with `scripts/eval.mjs` (before and after counts on real and generated images), and check the page in a browser.

## The AI-image model

`public/models/commfor-384-grok-ft-w8.onnx` is Community Forensics ViT-S/16 (384), fine-tuned on phone photos paired with Grok edits of them, exported with two outputs (`logits` and the 24 × 24 `token` map for the heatmap) and int8 weight-only quantisation. How and why, with the test results: `docs/finetune.md`.

`public/classify.js` holds `MODEL` (file, size, `threshold`). The line is 0.6. `scripts/eval.mjs` has its own `LINE`; keep the two equal.

The image score is the mean tile score over the 384-pixel grid at the image's own resolution (long side capped at 3,072). Heatmap tiles also run every 192 pixels. The fine-tuning scripts score images the same way; keep them in step if the tiling changes.

Retraining: `scripts/finetune/` (collect Commons photos, write edit jobs, prepare crops and replay shards, train, score), then `python scripts/export-model.py <checkpoint.pt> public/models/<name>.onnx`. Update `MODEL.url`, the SHA-256 in `THIRD_PARTY_NOTICES.md`, and the numbers in `README.md`, `public/index.html` (How detection works), `public/app.js` (the comparison note) and `docs/finetune.md` in the same change.

The export script writes its float32 intermediate as `export-fp32.onnx` in the working directory (gitignored). Never leave it in `public/`: everything there is deployed.

## Conventions

- Plain ES modules, no framework, no bundler. Runtime loads only ONNX Runtime Web and a font from CDNs; a new dependency needs a reason.
- Copy states what was measured, on what, and when. Numbers on the page and in the docs come from a run you can repeat.
- Test images and personal photos never go in the repository. Images with location data never leave the machine.
- Phase 2 conversion code must live in its own module(s) and must not be mixed into the public detection or metadata-cleaner paths.
```

You can copy the block above directly into `CLAUDE.md`.  

It keeps the original detector/cleaner philosophy intact while giving clear, bounded instructions for a Phase 2 “make it look more like a real camera photo” path.