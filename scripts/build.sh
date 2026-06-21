#!/usr/bin/env bash
set -e

# Defaults
NAME=""

# Parse args
for arg in "$@"; do
  case $arg in
  --name=*)
    NAME="${arg#*=}"
    shift
    ;;
  esac
done

# Determine build directory
if [ -z "$NAME" ]; then
  BUILD_DIR="build"
else
  BUILD_DIR="${NAME}-build"
fi

# git submodule update --init --progress --depth 1
cmake -Wno-dev -DENABLE_CLANG_TIDY=OFF -DCMAKE_BUILD_TYPE=Release -B "$BUILD_DIR" -G Ninja .
cmake --build "$BUILD_DIR"
