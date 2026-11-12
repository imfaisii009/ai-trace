# Third-party notices

## Community Forensics model weights

`public/models/commfor-384-grok-ft-w8.onnx` is the Community Forensics model
(https://github.com/JeongsooP/Community-Forensics, weights from
https://huggingface.co/OwensLab/commfor-model-384), fine-tuned by the AI Trace contributors
(docs/finetune.md), exported to ONNX with an extra per-patch output for the heatmap, its weights
stored as 8-bit integers, by `scripts/export-model.py`. Paper: Jeongsoo Park and Andrew Owens, "Community Forensics:
Using Thousands of Generators to Train Fake Image Detectors", CVPR 2025
(https://arxiv.org/abs/2411.04125).

SHA-256 of the bundled file: 15a482f6b1613b92538d27ed71ab1ab503e237a929a1e79e67892e5de034b228

```
MIT License

Copyright (c) 2025 Jeongsoo Park

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## ONNX Runtime Web

Loaded at run time from https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/ and not bundled.
Copyright (c) Microsoft Corporation, MIT licence
(https://github.com/microsoft/onnxruntime/blob/main/LICENSE).

## Fine-tuning data

The bundled model was fine-tuned on, and does not contain or redistribute:

- Photos from Wikimedia Commons, each under the licence on its file page, listed with
  their file pages in `docs/finetune-photos.txt`, and edits of 359 of them made with Grok's image editor.
- Images from the Community Forensics dataset (https://huggingface.co/datasets/OwensLab/CommunityForensics),
  CC BY 4.0, by Jeongsoo Park and Andrew Owens.
