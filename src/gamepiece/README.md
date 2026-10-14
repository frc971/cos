```sh
ssh root@dev-orin 'cd /root && ./tools/hsv_cluster_visualizer' && \
  mkdir -p build/fake_gamepiece-clusters && \
  rsync -avz root@dev-orin:/root/build/fake_gamepiece-clusters/ \
    build/fake_gamepiece-clusters/
```
