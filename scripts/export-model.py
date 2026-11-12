"""Export a Community Forensics checkpoint to the 8-bit ONNX model the page runs.

    python3 -m venv .venv && . .venv/bin/activate
    pip install torch timm safetensors huggingface_hub onnx onnxscript onnxruntime
    python scripts/export-model.py                      # published weights -> public/models/commfor-384-maps-w8.onnx
    python scripts/export-model.py ep5.pt out.onnx      # a fine-tuned checkpoint (scripts/finetune/train.py)

The page ships the fine-tuned model, public/models/commfor-384-grok-ft-w8.onnx (docs/finetune.md).

Tested with torch 2.15, timm 1.0.31, onnx 1.23 and onnxruntime 1.31. The model is a timm
vit_small_patch16_384 with a one-logit head. The graph has two outputs:

    logits  [1, 1]       the classifier's logit for "generated"
    token   [1, 24, 24]  the same head applied to each final patch token, through a sigmoid

`token` is what the page draws the heatmap from. Weights are then stored as 8-bit integers with one
scale per output channel and expanded back to float32 by DequantizeLinear nodes, so all arithmetic
stays float32: the file shrinks from 88 MB to 23 MB and the maps keep their shape. Over 31 tiles of
real and generated images the 8-bit token maps correlate with float32 at a median of 0.98 and tile
scores move by 0.007 on average. Dynamic quantisation, which also quantises activations, moved
tile scores by up to 0.66 and bent the Grad-CAM maps we tried first (median correlation 0.66).
"""
import os
import sys
import numpy as np
import onnx
import onnxruntime as ort
import timm
import torch
from huggingface_hub import hf_hub_download
from onnx import helper, numpy_helper
from safetensors.torch import load_file

weights = hf_hub_download('OwensLab/commfor-model-384', 'model.safetensors')
model = timm.create_model('vit_small_patch16_384', pretrained=False, num_classes=1)
model.load_state_dict(torch.load(sys.argv[1], map_location='cpu') if len(sys.argv) > 1 else {k.removeprefix('vit.'): v for k, v in load_file(weights).items()})
model.eval()
for b in model.blocks:
    b.attn.fused_attn = False  # the exporter has no ONNX form for the fused attention kernel


class Maps(torch.nn.Module):
    def forward(self, x):
        z = model.forward_features(x)
        return model.head(z[:, 0]), torch.sigmoid(model.head(z[:, 1:])[..., 0]).view(-1, 24, 24)


out = sys.argv[2] if len(sys.argv) > 2 else 'public/models/commfor-384-maps-w8.onnx'
os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
fp32 = 'export-fp32.onnx'  # intermediate, in the working directory (gitignored), never next to the page
x = torch.randn(1, 3, 384, 384)
torch.onnx.export(Maps(), (x,), fp32, input_names=['pixel_values'], output_names=['logits', 'token'], opset_version=18, dynamo=True)
with torch.no_grad():
    ref = [t.numpy() for t in Maps()(x)]
got = ort.InferenceSession(fp32).run(None, {'pixel_values': x.numpy()})
print('export check, float32 ONNX vs PyTorch: logit', float(np.abs(ref[0] - got[0]).max()), 'token map', float(np.abs(ref[1] - got[1]).max()))
m = onnx.load(fp32)  # also loads any external data file the exporter wrote alongside

# Weight-only int8, one scale per output channel.
g = m.graph
axis_of = {}
for n in g.node:
    if n.op_type == 'MatMul':
        axis_of[n.input[1]] = -1
    elif n.op_type == 'Gemm' and len(n.input) > 1:
        axis_of[n.input[1]] = 0 if next((a.i for a in n.attribute if a.name == 'transB'), 0) else 1
    elif n.op_type == 'Conv' and len(n.input) > 1:
        axis_of[n.input[1]] = 0
keep, dq = [], []
for init in g.initializer:
    w = numpy_helper.to_array(init)
    if init.name not in axis_of or w.dtype != np.float32 or w.size < 4096:
        keep.append(init)
        continue
    ax = axis_of[init.name] % w.ndim
    shape = [1] * w.ndim
    shape[ax] = -1
    scale = (np.abs(w).max(axis=tuple(i for i in range(w.ndim) if i != ax)) / 127).astype(np.float32)
    scale[scale == 0] = 1
    q = np.clip(np.round(w / scale.reshape(shape)), -127, 127).astype(np.int8)
    keep += [numpy_helper.from_array(q, init.name + '_q'), numpy_helper.from_array(scale, init.name + '_s')]
    dq.append(helper.make_node('DequantizeLinear', [init.name + '_q', init.name + '_s'], [init.name], axis=ax))
del g.initializer[:]
g.initializer.extend(keep)
nodes = list(g.node)
del g.node[:]
g.node.extend(dq + nodes)
onnx.checker.check_model(m)
onnx.save(m, out, save_as_external_data=False)
print(f'{len(dq)} weights quantised, {os.path.getsize(out) / 1e6:.1f} MB')
