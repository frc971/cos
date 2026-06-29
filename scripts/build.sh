#!/usr/bin/env bash
set -e

# Defaults
NAME=""
RUN_UNIT_TESTS=0

# Parse args
for arg in "$@"; do
  case $arg in
  --name=*)
    NAME="${arg#*=}"
    shift
    ;;
  --unit-tests | --run-unit-tests)
    RUN_UNIT_TESTS=1
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

if [ "$RUN_UNIT_TESTS" -eq 1 ]; then
  ctest --test-dir "$BUILD_DIR" --output-on-failure
fi
