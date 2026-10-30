# Ball U-Net

`ball_unet.onnx` is the fixed-shape grayscale segmentation model.
`ball_unet.engine` is its FP16 TensorRT engine, built and tested on dev-orin
(Orin, compute capability 8.7) with TensorRT 10.16.2. Rebuild the engine for
other GPUs or incompatible TensorRT versions using the ONNX model or
`scripts/export_unet_engine.sh`.

Input `image` and output `probability` both have shape `[1, 1, 400, 640]`.
The source checkpoint is
`yasencomp:/home/charlie/charlie_home/ball-segmentation/artifacts/training-grayscale/best.pt`,
with SHA-256 `0d2d33035a8618f521e435855dfa41f0a0dcc10e2e42adbd6ecb11e0399f1c1e`.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `ball_unet.engine` | 4,319,628 | `c55d6bd213679ba99b7748717b10d5717e3f6b4e267931a8fbd3f51782f9d28a` |
| `ball_unet.onnx` | 7,860,586 | `d1d655ce6bdcf8671187b3bac6ecbf25df0496fba831c9056f6e4abad8683bbf` |

Validation: ONNX shape checks and CPU output comparison against PyTorch passed.
On dev-orin, all three ball-pathing node tests (including GPU inference) and
five pathing logic tests passed. A 10-second TensorRT benchmark at 40W measured
6.15 ms mean latency and 166 inferences/second. These checks do not measure
segmentation accuracy on labeled images.

The tested engine is installed at `/root/models/ball_unet.engine` on dev-orin.
