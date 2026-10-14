#!/usr/bin/env bash
# Export a grayscale Ultralytics detector with NMS embedded in ONNX, then build
# a TensorRT engine. Run this script on the workstation: Python runs locally,
# while the Orin's existing trtexec builds the engine over SSH.
#
# Usage: export_yolo_engine.sh MODEL.pt [OUTPUT.engine]
# Both arguments are local paths. The engine is kept on the Orin and downloaded
# to OUTPUT.engine. ORIN_HOST defaults to root@dev-orin; ORIN_OUTPUT_DIR defaults
# to /root/gamepiece_models. PYTHON_BIN selects the local export environment.

set -Eeuo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 MODEL.pt [OUTPUT.engine]" >&2
  exit 2
fi

MODEL_PATH="$1"
MODEL_DIR="$(cd -- "$(dirname -- "${MODEL_PATH}")" 2>/dev/null && pwd)" || {
  echo "Model directory not found: $(dirname -- "${MODEL_PATH}")" >&2
  exit 1
}
MODEL_PATH="${MODEL_DIR}/$(basename -- "${MODEL_PATH}")"
MODEL_NAME="$(basename -- "${MODEL_PATH}")"
MODEL_STEM="${MODEL_NAME%.*}"
ENGINE_PATH="${2:-${MODEL_DIR}/${MODEL_STEM}_nms_gray.engine}"
ONNX_PATH="${ONNX_PATH:-${ENGINE_PATH%.*}.onnx}"
IMAGE_SIZE="${IMAGE_SIZE:-640}"
CONF_THRESHOLD="${CONF_THRESHOLD:-0.25}"
IOU_THRESHOLD="${IOU_THRESHOLD:-0.45}"
MAX_DETECTIONS="${MAX_DETECTIONS:-300}"
ONNX_OPSET="${ONNX_OPSET:-17}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ORIN_HOST="${ORIN_HOST:-root@dev-orin}"
ORIN_OUTPUT_DIR="${ORIN_OUTPUT_DIR:-/root/gamepiece_models}"

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "Model not found: ${MODEL_PATH}" >&2
  exit 1
fi

mkdir -p -- "$(dirname -- "${ONNX_PATH}")" "$(dirname -- "${ENGINE_PATH}")"
REMOTE_ENGINE_PATH="${ORIN_OUTPUT_DIR%/}/$(basename -- "${ENGINE_PATH}")"
REMOTE_ONNX_PATH="${REMOTE_ENGINE_PATH%.*}.onnx"
# Timing caches are specific to the device's TensorRT builder, so this path is
# remote, unlike ONNX_PATH and OUTPUT.engine.
TIMING_CACHE_PATH="${TIMING_CACHE_PATH:-${REMOTE_ENGINE_PATH%.*}.timing.cache}"

# SSH sends a command string to the remote shell. Quote paths explicitly rather
# than relying on local argument boundaries to survive that extra shell.
shell_quote() {
  printf "'%s'" "${1//\'/\'\\\'\'}"
}

echo "Exporting ${MODEL_PATH} to ${ONNX_PATH} on this workstation"
YOLO_AUTOINSTALL=false "${PYTHON_BIN}" - "${MODEL_PATH}" "${ONNX_PATH}" "${IMAGE_SIZE}" \
  "${CONF_THRESHOLD}" "${IOU_THRESHOLD}" "${MAX_DETECTIONS}" \
  "${ONNX_OPSET}" <<'PY'
import shutil
import sys
from pathlib import Path

try:
    import onnx
    import onnxslim
    import torch
    from ultralytics import YOLO
except ImportError as exc:
    raise SystemExit(
        f"Missing local Python dependency: {exc.name}. Install "
        "scripts/yolo-export-requirements.txt in a workstation virtual "
        "environment and select its Python with PYTHON_BIN."
    ) from exc

model_path = Path(sys.argv[1]).resolve()
onnx_path = Path(sys.argv[2]).resolve()
image_size = int(sys.argv[3])
conf = float(sys.argv[4])
iou = float(sys.argv[5])
max_det = int(sys.argv[6])
opset = int(sys.argv[7])

if image_size <= 0 or max_det <= 0:
    raise SystemExit("IMAGE_SIZE and MAX_DETECTIONS must be positive integers.")
if not 0.0 <= conf <= 1.0 or not 0.0 <= iou <= 1.0:
    raise SystemExit("CONF_THRESHOLD and IOU_THRESHOLD must be between 0 and 1.")

model = YOLO(str(model_path))
first_conv = next(
    (module for module in model.model.modules() if isinstance(module, torch.nn.Conv2d)),
    None,
)
if first_conv is None:
    raise SystemExit("The checkpoint contains no Conv2d layer.")
if first_conv.in_channels != 1:
    raise SystemExit(
        "Refusing to export a non-grayscale checkpoint: first convolution has "
        f"{first_conv.in_channels} input channels, expected 1."
    )

exported_path = Path(
    model.export(
        format="onnx",
        imgsz=image_size,
        batch=1,
        dynamic=False,
        nms=True,
        conf=conf,
        iou=iou,
        max_det=max_det,
        opset=opset,
        # Fold the NMS padding/shape graph before TensorRT consumes it. The
        # unsimplified graph can build but overflow during enqueue on Orin.
        simplify=True,
        device="cpu",
    )
).resolve()

if exported_path != onnx_path:
    shutil.move(str(exported_path), str(onnx_path))

graph = onnx.load(str(onnx_path), load_external_data=False)
input_dims = graph.graph.input[0].type.tensor_type.shape.dim
input_channels = input_dims[1].dim_value if len(input_dims) >= 2 else 0
if input_channels != 1:
    raise SystemExit(
        f"Exported ONNX input is not grayscale (channel dimension={input_channels})."
    )
if not any(node.op_type == "NonMaxSuppression" for node in graph.graph.node):
    raise SystemExit("Exported ONNX graph does not contain NonMaxSuppression.")

print(f"Verified grayscale input and embedded NMS in {onnx_path}")
PY

echo "Uploading ONNX to ${ORIN_HOST}:${REMOTE_ONNX_PATH}"
ssh "${ORIN_HOST}" \
  "mkdir -p -- $(shell_quote "${ORIN_OUTPUT_DIR}") && cat > $(shell_quote "${REMOTE_ONNX_PATH}")" \
  < "${ONNX_PATH}"

echo "Building FP16 TensorRT engine on ${ORIN_HOST}"
ssh "${ORIN_HOST}" \
  "bash -s -- $(shell_quote "${REMOTE_ONNX_PATH}") $(shell_quote "${REMOTE_ENGINE_PATH}") $(shell_quote "${TIMING_CACHE_PATH}")" <<'REMOTE'
set -Eeuo pipefail
onnx_path="$1"
engine_path="$2"
timing_cache_path="$3"

if command -v trtexec >/dev/null 2>&1; then
  trtexec_bin="$(command -v trtexec)"
elif [[ -x /usr/src/tensorrt/bin/trtexec ]]; then
  trtexec_bin=/usr/src/tensorrt/bin/trtexec
else
  echo "The Orin image must provide trtexec; add it through the Yocto build." >&2
  exit 1
fi

# Preserve a working engine if the builder fails.
pending_engine="${engine_path}.building.$$"
trap 'rm -f -- "$pending_engine"' EXIT
mkdir -p -- "$(dirname -- "$timing_cache_path")"
"${trtexec_bin}" \
  --onnx="${onnx_path}" \
  --saveEngine="${pending_engine}" \
  --timingCacheFile="${timing_cache_path}" \
  --fp16 \
  --skipInference

if [[ ! -s "${pending_engine}" ]]; then
  echo "trtexec completed without producing a non-empty engine: ${engine_path}" >&2
  exit 1
fi
mv -- "${pending_engine}" "${engine_path}"
REMOTE

pending_local_engine="$(mktemp "${ENGINE_PATH}.download.XXXXXX")"
trap 'rm -f -- "$pending_local_engine"' EXIT
ssh "${ORIN_HOST}" "cat -- $(shell_quote "${REMOTE_ENGINE_PATH}")" > "${pending_local_engine}"
if [[ ! -s "${pending_local_engine}" ]]; then
  echo "Downloaded engine is empty: ${ENGINE_PATH}" >&2
  exit 1
fi
mv -- "${pending_local_engine}" "${ENGINE_PATH}"
echo "Created ${ENGINE_PATH} and ${ORIN_HOST}:${REMOTE_ENGINE_PATH}"
