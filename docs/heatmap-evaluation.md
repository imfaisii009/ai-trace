# Heatmap evaluation

How the heatmap method was chosen, with the numbers. Run 11 November 2026 on the published Community Forensics ViT-S/16 (384) weights, before the page switched to the fine-tuned model ([finetune.md](finetune.md)). The map is computed the same way for both.

## What a heatmap can and cannot show

A heatmap explains a classifier's score; it does not add evidence the classifier did not find. So it was judged on two things: does it point at the region that is actually generated, when that is known, and does it point at what drives the model's score.

## Tests

1. **Pasted patches (known answer).** For each of 24 real photos (16 phone photos from Wikimedia Commons, 8 Kodak scans), a square of 448 or 512 pixels was cut from a generated image (Grok, ChatGPT, the mascot and two playground stills) and pasted at a random place, then saved as PNG and as JPEG quality 90: 48 composites, with the pasted pixels known. Measured per composite: AUROC of the map over the pasted mask (0.5 is chance, 1 is perfect), and the pointing game (is the hottest pixel inside the patch). "Detected" means the composites where the map's mean over the patch passed 0.3, that is where the model noticed the patch at all.
2. **Deletion (faithfulness).** Blur (Gaussian, radius 3) the 20% of pixels the map rates highest on a generated image and score the image again. A map that finds what drives the score should make it drop. Baseline: blurring random 32-pixel blocks. 11 images: 5 playground stills, a landscape, a poster, Grok, ChatGPT, the mascot, and the edited photo from the worked example.
3. **Real photos (false heat).** The map's mean, and the share of pixels above 0.5, on 8 Kodak scans and 8 phone photos.

## Methods compared

All maps are per 384-pixel tile, stitched over the image. The suffix `_xp` means the tile's map is divided by its own maximum and multiplied by the tile's score, so a tile the model calls real stays cold.

| Method | What it is | Cost per tile |
|---|---|---|
| tile | each tile's score, painted over the tile | forward |
| token head | the classifier head applied to each of the 24 × 24 final patch tokens, through a sigmoid | forward |
| gradient × input | gradient of the logit at the patch embedding, times the embedding, summed, ReLU | forward + backward |
| Grad-CAM | last block's normalised input, weighted by its mean gradient, ReLU | forward + backward |
| saliency | absolute input gradient | forward + backward |
| attention rollout | attention from the class token, multiplied through the layers (Abnar and Zuidema, 2020) | forward |

### Round 1: all methods, tiles every 384 pixels

Pasted-patch AUROC, mean over 48 composites (PyTorch, float32):

| tile | token head | gradient × input | Grad-CAM | saliency | rollout | rollout_xp | saliency_xp |
|---|---|---|---|---|---|---|---|
| 0.62 | 0.59 | 0.57 | 0.50 | 0.60 | 0.50 | 0.61 | 0.61 |

Raw gradient maps and rollout were near chance. Scaling by the tile score (`_xp`) helped every method. The finalists went to round 2.

### Round 2: finalists, tiles every 192 pixels

PyTorch, float32, images capped at 1,536 pixels:

| Method | AUROC, all 48 | AUROC, detected (19) | Pointing | Deletion: score after blur (base 0.305, random 0.404) | Real: map mean | Real: share > 0.5 |
|---|---|---|---|---|---|---|
| tile | 0.750 | 0.843 | 16/48 | 0.267 | 0.030 | 1.9% |
| **token head_xp** | **0.746** | **0.831** | **22/48** | **0.230** | **0.022** | **1.4%** |
| gradient × input_xp | 0.699 | 0.774 | 25/48 | 0.172 | 0.002 | 0.0% |
| Grad-CAM_xp | 0.637 | 0.706 | 19/48 | 0.193 | 0.008 | 0.1% |
| token head (raw) | 0.665 | 0.765 | 12/48 | 0.211 | 0.110 | 5.7% |

Every map beat random blurring on all 11 deletion images.

**Choice: token head_xp.** It finds pasted regions as well as the tile map does (AUROC 0.75 against 0.75) at 16-pixel instead of 384-pixel resolution, and better than either gradient map (0.70 and 0.64). Gradient × input is more faithful on the deletion test and runs colder on real photos, but needs a backward pass, about three times the cost per tile, and ONNX Runtime Web has no autograd, so it would have to be exported as a second graph. The raw token head paints real photos hot (5.7% of pixels above 0.5); scaling by the tile score fixes that.

### Round 3: the shipped model, 384 against 192

The exported ONNX model with 8-bit weights, at the page's settings (images capped at 3,072 pixels), token head_xp:

| Tiles every | AUROC, all 48 | AUROC, detected | Deletion (base 0.423, random 0.524) | Real: share > 0.5 |
|---|---|---|---|---|
| 384 pixels | 0.620 | 0.702 (7) | 0.325, 10/11 beat random | 2.2% |
| 192 pixels | 0.739 | 0.877 (8) | 0.338, 11/11 beat random | 1.1% |

Overlap is worth the cost: the page runs tiles every 192 pixels for the map. The image score still comes from the tiles on the 384-pixel grid (a subset of the same run), which is what the score tests were run on.

### Round 4: stitching overlapping tiles

Plain mean against tent weights (1 at a tile's centre, 0 at its edge), same 192-pixel tiles, shipped model:

| Stitching | AUROC, all 48 | AUROC, detected (8) | Pointing | Real: share > 0.5 |
|---|---|---|---|---|
| plain mean | 0.739 | 0.877 | 18/48 | 1.1% |
| tent | 0.736 | 0.883 | 22/48 | 1.5% |

A tie on accuracy (tent won 22 composites, lost 24). The page uses tent because plain averaging leaves visible steps at every 192-pixel tile edge, which look like findings and are not.

## Fidelity of the 8-bit model

Over 31 tiles of real and generated images, against float32: token maps correlate at a median of 0.979 (lowest 0.959), mean absolute map difference 0.022, tile score difference 0.007 on average (largest 0.086). Dynamic quantisation, which also quantises activations, moved tile scores by up to 0.66 and was dropped.

## Limits found

- **Small pasted regions are invisible to this model.** Patches of 192 to 320 pixels left the image score at about 3%. Patches of 448 to 512 pixels were detected in about a quarter of composites. Community Forensics recognises whole generated or regenerated images far better than a generated region inside a real photo, and no map can point at what the model does not see.
- **Position matters.** In the worked example a tile over the carpet scored 87% while the tiles 192 pixels to either side scored 10% and 1%. The model's answer for a patch depends on what else is in the crop.
- **The set is small.** 48 composites, 11 deletion images, 16 real photos. The numbers rank the methods; they are not accuracy figures for the wild.
