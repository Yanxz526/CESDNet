# Optional legacy ShiftWise operator

Only the legacy `profile_a` Potsdam teacher requires this extension.
The default `paper_v1` model does not import it.

Install a CUDA toolkit matching the installed PyTorch CUDA version, including
`nvcc`, and a compatible C++ compiler. Build using the Python environment
used to run the model:

```bash
cd toolbox/models/shift_wiseConv_master/shiftadd
# Set CUDA_HOME if the matching toolkit is not discovered automatically.
MAX_JOBS=2 python -m pip install --no-build-isolation .
python -c 'import addshift, addshiftmp, addshiftmp_linear, addshiftmp_em, addshiftmp_blur'
```

The build targets the available GPU architecture by default. For cross-compilation,
set `TORCH_CUDA_ARCH_LIST` to the target GPU's compute capability.
Upstream licenses accompany the operator sources.
