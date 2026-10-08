#!/usr/bin/env bash
source /orin-sdk/environment-setup-aarch64-oe4t-linux && cmake -S . -B build -G Ninja -Wno-dev -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && cmake --build build --parallel
