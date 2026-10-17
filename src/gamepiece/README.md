The visualizer defaults to `constants/gamepiece/camera.json` (translations in
meters, rotations in degrees) and `/cos-logs/gamepiece/110.600000.jpg` on dev-orin.
Build and deploy with `cmake --build build --target hsv_cluster_visualizer` and
`cmake --build build --target dev-orin`, then run from the repository root:

```sh
ssh root@dev-orin 'cd /root && ./tools/hsv_cluster_visualizer' && \
  mkdir -p build/fake_gamepiece-clusters && \
  rsync -avz root@dev-orin:/root/build/fake_gamepiece-clusters/ \
    build/fake_gamepiece-clusters/
```
