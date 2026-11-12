# Roadmap

Open work, grouped by area. Each item says what done looks like. Pick one, open an issue saying you are on it, and send a pull request.

Ground rules for every change:

- Everything stays in the browser. No uploads, no servers, no analytics.
- A change to the model, the tiles or the 60% line comes with before and after counts from `scripts/eval.mjs` on real and generated images.
- New runtime dependencies need a reason in the pull request. The page currently loads only ONNX Runtime Web and a font.
- The project detects and explains. Changes whose purpose is to make generated images pass detectors are out of scope.

## Heatmap

The current map and the methods it beat are in [docs/heatmap-evaluation.md](docs/heatmap-evaluation.md). Any change to the map comes with that evaluation re-run.

- [ ] **Gradient × input as a second map.** It was the most faithful on the deletion test but needs a backward pass. Export it as its own ONNX graph (`torch.fx` `make_fx` records the backward pass, and the dynamo exporter converts it) and offer it as a slower, sharper option. Done: a toggle, timing on a 12 MP photo, and the evaluation re-run on the exported graph.
- [ ] **Edits, not pasted squares.** The evaluation pastes generated squares into real photos. Real edits are inpainted and blended. Build a set from an open inpainting model with the masks kept, and re-run the evaluation on it.
- [ ] **Colour-blind friendly scale.** The map uses jet, like most detectors. Offer a second scale (for example viridis).
- [ ] **See the photo under the map.** A control to fade or hide the map.
- [ ] **Export the heatmap.** Save the image with its map as a PNG.

## The AI-image model

- [ ] **Edits from more tools.** The model was fine-tuned on Grok edits only ([docs/finetune.md](docs/finetune.md)). Add edits of the same Commons photos from other editors (Gemini, ChatGPT, Firefly, open inpainting models) with `scripts/finetune/`, retrain, and report recall per editor. Done: per-editor recall before and after, and no rise in false positives on phone photos.
- [ ] **More kinds of real image.** Lossless film scans are the weakest real set (5 of 24 over the line). Add real scans, screenshots and photos from dedicated cameras to the fine-tuning data. Done: the Kodak row in docs/finetune.md at 0 without losing edit recall.
- [ ] **A public evaluation set.** A manifest (URL, licence, label, generator) of real photos and generated images, a script that downloads them, and a results table in the repo. No images committed. Done: at least 200 real and 200 generated, spread across phones, cameras, scans and generators.
- [ ] **Calibrate the line.** With the set above, plot precision and recall against the threshold and pick the line from it. Report recall per generator.
- [ ] **A second model.** Run another open detector next to Community Forensics (for example a CLIP-based one such as UnivFD) and show where they agree. Done: both scores and both maps, with eval numbers for each and for the pair.
- [ ] **WebGPU.** Try ONNX Runtime Web's WebGPU execution provider with WASM as the fallback. Done: timings for both on the same machine.
- [ ] **Smaller download.** Try fp16 or 4-bit weights. Done: size, speed and eval numbers against the current 23 MB model.
- [ ] **Known false positives.** Real photos that score high (heavy phone processing, small re-saved copies, scanned film). Collect examples and document the pattern on the page.

## Formats

- [ ] **HEIC and AVIF.** Walk ISOBMFF boxes, find EXIF, XMP and C2PA (the C2PA `uuid` box), and clean them. Score them with the model where the browser can decode them.
- [ ] **JPEG XL, GIF and TIFF.** Block walkers for each.
- [ ] **Video and audio.** C2PA and metadata in MP4/MOV, and ID3 and RIFF chunks in audio. Reading only to start with.
- [ ] **PDF and Office files.** Producer, creator and custom properties that name AI tools.
- [ ] **Rotation in PNG and WebP.** JPEG keeps its rotation in the clean copy; do the same for an `eXIf` chunk in PNG and the EXIF chunk in WebP.

## Provenance and watermarks

- [ ] **Verify C2PA.** Check the COSE signature against the certificate chain, check the hash assertions against the file, and say whether the signer is on the C2PA trust list. Today the fields are read, not verified.
- [ ] **Adobe TrustMark.** The decoder is open source and has ONNX models. Run it in the browser and report a found watermark.
- [ ] **Meta Stable Signature and Video Seal.** Both have public code and weights. Same as TrustMark.
- [ ] **SynthID.** No public detector exists. Keep pointing to Google's tools and update the page if that changes.

## Text

- [ ] **More chat-app artifacts.** Citation and link formats from Claude, Gemini, Perplexity, Copilot and Grok, each with a real example in the pull request.
- [ ] **File types.** Read text out of `.docx` and `.pdf` so their hidden characters can be checked.

## The page

- [ ] **Batch mode.** Drop a folder and get one row per file with its signals and score.
- [ ] **JSON report.** Download everything the image report shows as JSON.
- [ ] **Accessibility pass.** Keyboard use of the tabs and drop zone, screen-reader labels on the heatmap.
