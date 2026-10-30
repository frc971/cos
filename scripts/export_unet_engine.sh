#!/usr/bin/env bash
# Export the trained U-Net to ONNX, build its FP16 TensorRT engine on
# vision-bot, and download a copy. With no checkpoint argument the known model
# is streamed from yasencomp.
#
# Usage: scripts/export_unet_engine.sh [CHECKPOINT.pt] [OUTPUT.engine]
# Set INPUT_MODE=rgb for the pinned three-channel model (default: grayscale).

set -Eeuo pipefail

if [[ $# -gt 2 ]]; then
  echo "Usage: $0 [CHECKPOINT.pt] [OUTPUT.engine]" >&2
  exit 2
fi

INPUT_MODE="${INPUT_MODE:-grayscale}"
case "${INPUT_MODE}" in
  grayscale)
    EXPECTED_SHA256=0d2d33035a8618f521e435855dfa41f0a0dcc10e2e42adbd6ecb11e0399f1c1e
    default_checkpoint=/home/charlie/charlie_home/ball-segmentation/artifacts/training-grayscale/best.pt
    default_engine=ball_unet.engine
    ;;
  rgb)
    EXPECTED_SHA256=dcced817d26316f46f6c324a90ccc796849381f5b59ad28a5697efe44022436e
    default_checkpoint=/home/charlie/charlie_home/ball-segmentation/artifacts/training/best.pt
    default_engine=ball_unet_rgb.engine
    ;;
  *)
    echo "INPUT_MODE must be grayscale or rgb" >&2
    exit 2
    ;;
esac
SOURCE_HOST="${SOURCE_HOST:-yasencomp}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${default_checkpoint}}"
ORIN_HOST="${ORIN_HOST:-root@10.99.72.11}"
REMOTE_ENGINE="${REMOTE_ENGINE:-/root/gamepiece_models/${default_engine}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_ENGINE="${2:-${default_engine}}"
ONNX_PATH="${ONNX_PATH:-${OUTPUT_ENGINE%.*}.onnx}"

temporary_checkpoint=""
cleanup() {
  if [[ -n "${temporary_checkpoint}" ]]; then
    rm -f -- "${temporary_checkpoint}"
  fi
}
trap cleanup EXIT

if [[ $# -ge 1 ]]; then
  checkpoint="$1"
else
  temporary_checkpoint="$(mktemp /tmp/ball-unet-checkpoint.XXXXXX.pt)"
  echo "Downloading checkpoint from ${SOURCE_HOST}:${SOURCE_CHECKPOINT}"
  ssh "${SOURCE_HOST}" "cat -- '${SOURCE_CHECKPOINT}'" > "${temporary_checkpoint}"
  checkpoint="${temporary_checkpoint}"
fi

if [[ ! -f "${checkpoint}" ]]; then
  echo "Checkpoint not found: ${checkpoint}" >&2
  exit 1
fi
actual_sha256="$(sha256sum "${checkpoint}" | awk '{print $1}')"
if [[ "${actual_sha256}" != "${EXPECTED_SHA256}" ]]; then
  echo "Unexpected checkpoint SHA-256: ${actual_sha256}" >&2
  echo "Expected: ${EXPECTED_SHA256}" >&2
  exit 1
fi

mkdir -p -- "$(dirname -- "${ONNX_PATH}")" "$(dirname -- "${OUTPUT_ENGINE}")"
echo "Exporting fixed-shape ONNX to ${ONNX_PATH}"
"${PYTHON_BIN}" - "${checkpoint}" "${ONNX_PATH}" "${INPUT_MODE}" <<'PY'
import sys
from pathlib import Path

try:
    import onnx
    import onnxruntime
    import torch
    from torch import nn
except ImportError as exc:
    raise SystemExit(
        f"Missing Python dependency {exc.name!r}; install "
        "scripts/unet-export-requirements.txt and set PYTHON_BIN."
    ) from exc


class DoubleConv(nn.Sequential):
    def __init__(self, inputs, outputs):
        super().__init__(
            nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
            nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        widths = [16, 32, 64, 128, 256]
        self.down = nn.ModuleList(
            DoubleConv(i, o) for i, o in zip([input_channels] + widths[:-1], widths)
        )
        self.up = nn.ModuleList(
            DoubleConv(i + o, o) for i, o in zip(widths[:0:-1], widths[-2::-1])
        )
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, value):
        skips = []
        for block in self.down[:-1]:
            value = block(value)
            skips.append(value)
            value = torch.nn.functional.max_pool2d(value, 2)
        value = self.down[-1](value)
        for block, skip in zip(self.up, reversed(skips)):
            value = torch.nn.functional.interpolate(
                value, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
            value = block(torch.cat([value, skip], dim=1))
        return self.head(value).sigmoid()


checkpoint_path, output_path = map(Path, sys.argv[1:3])
input_mode = sys.argv[3]
input_channels = {"grayscale": 1, "rgb": 3}[input_mode]
saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
config = saved.get("config", {})
# Original RGB checkpoints predate input-mode metadata.
if (config.get("input_channels", 3) != input_channels or
        config.get("input_mode", "rgb") != input_mode):
    raise SystemExit(f"Checkpoint is not the expected {input_mode} model")
model = UNet(input_channels).eval()
model.load_state_dict(saved["model"], strict=True)
sample = torch.zeros(1, input_channels, 400, 640)
with torch.inference_mode():
    expected = model(sample)
torch.onnx.export(
    model,
    sample,
    output_path,
    input_names=["image"],
    output_names=["probability"],
    opset_version=17,
    do_constant_folding=True,
    dynamo=False,
)
graph = onnx.load(output_path)
onnx.checker.check_model(graph)
input_dims = [dimension.dim_value for dimension in
              graph.graph.input[0].type.tensor_type.shape.dim]
output_dims = [dimension.dim_value for dimension in
               graph.graph.output[0].type.tensor_type.shape.dim]
if input_dims != [1, input_channels, 400, 640] or output_dims != [1, 1, 400, 640]:
    raise SystemExit(f"Unexpected ONNX shapes: input={input_dims}, output={output_dims}")
if not torch.isfinite(expected).all() or expected.min() < 0 or expected.max() > 1:
    raise SystemExit("Export source produced invalid probabilities")
session = onnxruntime.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
actual = session.run(["probability"], {"image": sample.numpy()})[0]
torch.testing.assert_close(torch.from_numpy(actual), expected, rtol=1e-4, atol=1e-5)
print(f"Verified {output_path}: {input_dims} -> {output_dims}")
PY

remote_onnx="${REMOTE_ENGINE%.*}.onnx"
remote_pending="${REMOTE_ENGINE}.building"
echo "Uploading ONNX and building TensorRT engine on ${ORIN_HOST}"
ssh "${ORIN_HOST}" "mkdir -p -- '$(dirname -- "${REMOTE_ENGINE}")' && cat > '${remote_onnx}'" < "${ONNX_PATH}"
ssh "${ORIN_HOST}" "bash -s -- '${remote_onnx}' '${remote_pending}' '${REMOTE_ENGINE}'" <<'REMOTE'
set -Eeuo pipefail
onnx_path="$1"
pending="$2"
engine="$3"
trap 'rm -f -- "$pending"' EXIT
if command -v trtexec >/dev/null 2>&1; then
  trtexec_bin="$(command -v trtexec)"
elif [[ -x /usr/src/tensorrt/bin/trtexec ]]; then
  trtexec_bin=/usr/src/tensorrt/bin/trtexec
else
  echo "vision-bot does not provide trtexec" >&2
  exit 1
fi
"${trtexec_bin}" --onnx="${onnx_path}" --saveEngine="${pending}" \
  --fp16 --skipInference
test -s "${pending}"
mv -- "${pending}" "${engine}"
REMOTE

pending_local="$(mktemp "${OUTPUT_ENGINE}.download.XXXXXX")"
trap 'rm -f -- "${pending_local}"; cleanup' EXIT
ssh "${ORIN_HOST}" "cat -- '${REMOTE_ENGINE}'" > "${pending_local}"
test -s "${pending_local}"
mv -- "${pending_local}" "${OUTPUT_ENGINE}"
echo "Installed ${ORIN_HOST}:${REMOTE_ENGINE} and created ${OUTPUT_ENGINE}"
