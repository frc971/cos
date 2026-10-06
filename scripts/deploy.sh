#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

git submodule update --init --recursive
./scripts/build.sh
# build.sh runs in a child shell, so load the SDK here for the deploy target too.
source /opt/orin-sdk/environment-setup-aarch64-oe4t-linux
cmake --build build --target second-bot
rsync -avz --delete "$repo_root/constants/" root@10.9.71.11:/root/constants/
