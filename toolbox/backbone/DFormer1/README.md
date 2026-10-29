# DFormer-Tiny

Backbone source based on [VCIP-RGBD/DFormer](https://github.com/VCIP-RGBD/DFormer/blob/main/models/encoders/DFormer.py).
The upstream MIT license is retained as `LICENSE`.

Stage dimensions are `[32, 64, 128, 256]`. The student calls
`backbone(optical, x_e=None)`; the auxiliary stream derives from the first
optical channel. Pretrained initialization requires an explicit compatible
DFormer-Tiny checkpoint.
