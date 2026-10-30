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

## RGB variant

`ball_unet_rgb.onnx` and `ball_unet_rgb.engine` contain the three-channel
RGB model. Input `image` has shape `[1, 3, 400, 640]`, with RGB planes in
channel-first order and float32 pixel values scaled to [0, 1]. Output
`probability` has shape `[1, 1, 400, 640]`.

The source checkpoint is
`yasencomp:/home/charlie/charlie_home/ball-segmentation/artifacts/training/best.pt`,
with SHA-256 `dcced817d26316f46f6c324a90ccc796849381f5b59ad28a5697efe44022436e`.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `ball_unet_rgb.engine` | 4,297,692 | `06d84a6ca845e6216cef905e25df9a8ce1ce11085d5e03a2adaf8ad14845ed60` |
| `ball_unet_rgb.onnx` | 7,861,738 | `ac28477e6d6944346a132e7b9a2c2d431e8f1563f078002f21b4b4fa65945b1c` |

The RGB engine was built and tested on the same dev-orin GPU and TensorRT
version as the grayscale engine. ONNX shape checks and CPU output comparison
against PyTorch passed. A 10-second TensorRT inference benchmark at 40W
measured 6.31 ms mean latency and 165 inferences/second. Segmentation accuracy
on labeled images has not been measured here. The engine is installed at
`/root/models/ball_unet_rgb.engine` on dev-orin.

Regenerate the RGB artifacts with `INPUT_MODE=rgb scripts/export_unet_engine.sh`.
The default local output names are `ball_unet_rgb.onnx` and
`ball_unet_rgb.engine`. `ORIN_HOST` selects the engine build host, and
`REMOTE_ENGINE` selects its remote engine path. The grayscale export remains
the default when `INPUT_MODE` is unset.

Second-bot deploys `ball_unet_rgb.engine` by default, installing it as
`/root/models/ball_unet.engine`. It decodes the color camera to interleaved RGB
and preprocesses it into three normalized float planes. The node receives
`UnetInputMode::kRgb` explicitly in its constructor and validates the engine's
input shape. Grayscale remains supported with `UnetInputMode::kGrayscale` and
the matching engine. Second-bot and disk replay expose this selection through
`--ball_pathing_input_mode=rgb|grayscale` (default: `rgb`).
