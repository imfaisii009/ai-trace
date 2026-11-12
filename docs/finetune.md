# Fine-tuning the AI-image model on AI-edited photos

The page runs `public/models/commfor-384-grok-ft-w8.onnx`: Community Forensics ViT-S/16 (384), fine-tuned on 12 November 2026 so that it sees a whole-photo AI edit, not only a fully generated image, and still sees it after ordinary camera-style processing. This note covers why, how, the numbers, and how to repeat it.

## Why

The published Community Forensics weights recognise fully generated images well: 96% of a held-out set of Firefly 2, Imagen 3 and DALL·E 3 images at the old 20% line. They are much weaker on the case people bring to a detector most often, a real phone photo passed through an image model with "keep everything the same" or "make it look professional". At the old line they caught 58% of those edits and the heatmap lit only a few tiles of each, while they also called 19% of held-out phone photos AI. None of the ten other openly licensed detectors we benchmarked did better. The model had not been trained on regenerated photos.

## Data

- **Real photos.** 596 photos from Wikimedia Commons phone-camera categories (Google Pixel, Samsung Galaxy, iPhone, Xiaomi, OnePlus, Huawei, Motorola, Oppo, Sony Xperia, Nokia, Honor, Vivo, Realme, LG), at least 1,500 pixels on the short side, at the largest size Commons serves (1,280, 1,920 or 3,840 pixels). `scripts/finetune/commons.py` collects them. They are listed in [finetune-photos.txt](finetune-photos.txt); each keeps its own licence on Commons.
- **AI edits.** 495 of those photos, each edited twice by Grok's image editor (grok-imagine-image, output about one megapixel), 990 edits in all, made on 12 November 2026. The 14 prompts are in `scripts/finetune/split_jobs.py`: "Recreate this exact photo", "Enhance this photo while keeping everything the same", and twelve "same photo, same framing, but…" variations (sharper, golden light, overcast, distractions removed, tidied, evening lights, wet after rain, one object added, a colour changed, background details replaced, professionally edited, more vivid).
- **Replay.** 2,400 generated and 3,692 real images (every real image in the evaluation shard) from the [Community Forensics dataset](https://huggingface.co/datasets/OwensLab/CommunityForensics) (CC BY 4.0): Gansformer, Firefly 2 and 3, FLUX schnell and dev, Ideogram, DALL·E 3, and real photos from its evaluation shard. They keep the model from forgetting what it already knew.

Photos are split into training and validation by a hash of the photo, so a photo and both of its edits always land on the same side. The test sets below share no image with training; 19 generated test images that also appeared in the replay set were removed from the test.

## Training

`scripts/finetune/train.py`, on one rented GPU, about ten minutes:

- 384-pixel crops: ten native-resolution crops per real photo, the whole photo resized, and random crops of each edit and replay image.
- Both classes get the same random downscaling (40-100% for whole photos, 60-100% half the time for the rest) and the same JPEG re-encoding (quality 70-98, 60% of the time), so neither size nor compression gives the label away.
- Both classes also get camera-style processing, each step 30% of the time: lateral chromatic aberration (up to 0.3%), sensor grain (luminance sigma 1-6 of 255, chroma 0-3) and sharpening. Each of these alone took AI edits under the line before; see Robustness.
- AdamW, learning rate 3e-5, weight decay 0.05, cosine schedule with warm-up, batch 64, bfloat16, 6 epochs. Edits are repeated 8 times per epoch and replay 4 times, to balance the classes.
- Epoch 5 of 6 was kept: epochs 4, 5 and 6 all caught every test edit, and epoch 5 called the fewest of the 329 real test images AI (5, against 8 and 7).

This is the third version. The second, from 11 November 2026, was trained the same way without the camera-style processing (494 photos, 987 edits). On the test sets below it caught the same images and called 10 of the 329 real images AI, 5 of them film scans, but grain, sharpening, chromatic aberration and JPEG applied together took every test edit under its line. Its training edits were not kept, so this version's were made again from the same photos and prompts. The first, also from 11 November 2026 (460 photos, 717 edits, 1,500 replay reals, 8 epochs, line 70%), caught 99 of 102 edits and called 5 of 254 real images AI, 3 of them film scans. It was replaced the same day after an AI edit of a room photo scored 47% on the page, under its line; the second version scored that photo 70%.

## Results

Scored on 12 November 2026 with the page's settings: tiles at the image's own resolution (capped at 3,072 pixels), mean over the 384-pixel grid. "Called AI" counts images at or over the line. All three models were scored on the same images. The Grok test edits are two edits of each of 51 held-out phone photos, made on 12 November 2026 with the same prompts as the training edits.

| Test set | Images | Published weights, line 20% | Second version (11 November), line 60% | This version, line 60% |
|---|---|---|---|---|
| Grok edits of held-out phone photos | 102 | 60 (59%) | 102 (100%) | 102 (100%) |
| Grok text-to-image photos (rooms, streets, food, kitchens, parks) | 20 | 12 | 20 | 20 |
| Community Forensics held-out generated images | 101 | 99 | 101 | 101 |
| Held-out phone photos from Commons | 80 | 7 | 0 | 0 |
| Community Forensics held-out real images | 150 | 1 | 1 | 2 |
| Kodak test photos (lossless film scans) | 24 | 0 | 5 | 1 |
| The same Kodak photos saved as JPEG (quality 90) | 24 | 0 | 2 | 1 |
| The source photos of the held-out edits, at the edit's size | 51 | 0 | 2 | 1 |

Across the three main real sets (phone photos, Community Forensics real, Kodak) this version calls 3 of 254 images AI, against 6 for the second version and 8 for the published weights. Ranked by score (AUROC, 0.5 is chance), Grok edits against those real images rise from 0.92 with the published weights to 0.997. The margin is smaller than before: the lowest-scoring test edit scores 0.67, against 0.89 with the second version.

The heatmap changed with it. On the Grok test edits, 98% of grid tiles score 0.5 or more (972 of 988), against 30% with the published weights, so the map covers the image instead of a few tiles. On a real phone photo with a camera-app stamp (date, place and compass drawn on by the app), the second version put the heat on the stamp band, which is rendered graphics, and kept the room cool, scoring the photo 20%. That photo was not kept, so this version has not been scored on it.

## Robustness

`scripts/robustness.mjs` applies mild camera-style steps to an image, one at a time and then all six in a chain, and scores every variant with the page's settings. On 12 November 2026 it ran on one edit of each of the 51 held-out test photos and on 30 held-out phone photos. "Over the line" counts images at or over 60%.

| Step | Edits over the line, second version | Edits over the line, this version |
|---|---|---|
| None | 51 | 51 |
| Grain (luminance sigma 4, chroma 2) | 41 | 49 |
| Wide, gentle sharpening | 50 | 48 |
| Warm white-balance drift | 51 | 51 |
| Chromatic aberration (0.15%) | 32 | 49 |
| Vignette | 51 | 51 |
| JPEG, quality 85 | 51 | 51 |
| All six in a chain | 0 | 36 |

None of the 30 phone photos went over the line under any step, with either version; the highest any of them scored was 0.32 with this version. The chain still takes 15 of the 51 edits under the line, so a deliberate post-process can still hide an edit.

The 60% line was chosen by looking at these test sets, so the false-positive rates above are a little optimistic. A separate calibration set is on the roadmap.

## Limits

- **Film scans.** Lossless scans of film (Kodak's test set) remain a weak spot: 1 of 24 over the line, 1 of 24 saved as JPEG. Scans, screenshots and other non-camera images are less well covered than phone photos.
- **One editor.** Every training edit came from Grok. Edits from other tools (Gemini, ChatGPT, Firefly, Photoshop generative fill) were not in training; the replay set covers their fully generated images, not their edits.
- **Small pastes.** A small generated region pasted into a real photo is still hard to see (see [heatmap-evaluation.md](heatmap-evaluation.md), measured on the published weights). The training edits regenerate the whole frame.
- **Camera-style processing.** Grain, sharpening, white-balance drift, chromatic aberration, vignetting and JPEG applied together still take 15 of 51 test edits under the line (see Robustness). Stronger settings and other steps, such as denoising, were not tested.
- **A score is a pointer.** It is evidence to weigh, not proof.

## Repeat it

```bash
cd scripts/finetune
pip install torch timm safetensors huggingface_hub pandas pyarrow pillow
python commons.py commons.json 1300                     # phone photos from Commons
python split_jobs.py commons.json jobs 500 6            # edit jobs: source URL + prompt
# Edit each job's photo with the image editor you want to cover. Save the results as
# data/edits/<sha1>_<k>.jpg and list them in data/edits.json as {"<sha1>": ["<sha1>_0.jpg", ...]}.
cp commons.json data/
python prepare.py data                                  # crops, replay shards, data/manifest.json
EPOCHS=6 python train.py data out                       # checkpoints out/ep1.pt ... and out/log.json
python eval_local.py sets.json scores.json out/ep5.pt   # test sets as {"name": {"label": 0 or 1, "files": [...]}}
python ../export-model.py out/ep5.pt ../../public/models/commfor-384-grok-ft-w8.onnx
```

Export with the library versions listed in `export-model.py`. On 12 November 2026 an export on the GPU box's own versions quantised no weights and wrote an 89 MB file; the script now stops when that happens. The page's model is about 23 MB.

`prepare.py` downloads the photos it does not have. Wikimedia rate-limits downloads hard, so fetch the served thumbnails slowly from one address. Edits from more tools, and real images from more cameras, scanners and screenshots, are the most useful contributions; see the roadmap.
