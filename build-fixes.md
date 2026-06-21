# Build Fixes

Two issues prevent `./scripts/build.sh` from working in a fresh container.

---

## 1. allwpilib cmake symlinks

### Problem
allwpilib was built in-tree at `/allwpilib/allwpilib/build-cmake/` but `cmake --install` was never run.
The `*-config.cmake` files in that directory reference sibling files like `wpiutil.cmake`, but the actual
files live one level deeper (`wpiutil/wpiutil.cmake`, `wpimath/wpimath.cmake`, etc.).

### Fix (run once per container)
```bash
cd /allwpilib/allwpilib/build-cmake
for lib in wpiutil wpimath ntcore hal cscore cameraserver datalog wpinet wpilibc commandsv2; do
  [ -f "$lib/$lib.cmake" ] && [ ! -e "$lib.cmake" ] && ln -sf "$lib/$lib.cmake" "$lib.cmake"
done
[ -f romiVendordep/romivendordep.cmake ] && [ ! -e romivendordep.cmake ] && ln -sf romiVendordep/romivendordep.cmake romivendordep.cmake
[ -f xrpVendordep/xrpvendordep.cmake ]  && [ ! -e xrpvendordep.cmake ]  && ln -sf xrpVendordep/xrpvendordep.cmake  xrpvendordep.cmake
[ -f apriltag/apriltag.cmake ]          && [ ! -e apriltag.cmake ]       && ln -sf apriltag/apriltag.cmake           apriltag.cmake
```

### Permanent fix (Dockerfile)
Add the block above after the allwpilib `cmake --build` step, or run a proper install:
```dockerfile
RUN cmake --install /allwpilib/allwpilib/build-cmake --prefix /allwpilib/allwpilib/build-cmake
```

---

## 2. NvBufSurface.h — case-insensitive filesystem (macOS host)

### Problem
`/cos` is bind-mounted from a macOS filesystem, which is case-insensitive.
`NvBufSurface.h` (the C++ wrapper class) included the system C API headers with quoted paths:

```cpp
#include "nvbufsurface.h"      // resolves to NvBufSurface.h itself on macOS — circular
#include "nvbufsurftransform.h"
```

When any translation unit included `"nvbufsurface.h"` (lowercase), the compiler resolved it to
the local `NvBufSurface.h` wrapper. The wrapper then tried to include `"nvbufsurface.h"` again,
which the include guard blocked, so the system types (`NvBufSurface`, `NvBufSurfaceColorFormat`, etc.)
were never defined.

### Fix (already applied to source)
Changed the two includes in `third_party/nvjpeg/NvBufSurface.h` to angle brackets, which skip
the current directory and search `-I` paths only:

```cpp
#include <nvbufsurface.h>       // finds /usr/src/jetson_multimedia_api/include/nvbufsurface.h
#include <nvbufsurftransform.h> // finds /cos/third_party/nvjpeg/nvbufsurftransform.h
```

This change is committed to the repo and requires no action in the Dockerfile.
