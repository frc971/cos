#!/usr/bin/env bash
set -euo pipefail

# Compile the stub independently of CMake and the robot project. Fetch only the
# two official WPILib Java libraries and their desktop JNI runtime dependencies.
version=2026.2.1
test_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cache=${COS_JAVA_TEST_CACHE:-/tmp/cos-cluster-java}
port=${COS_NT_TEST_PORT:-15810}
mkdir -p "$cache/classes" "$cache/native"
for artifact in ntcore wpiutil wpinet; do
  base="https://frcmaven.wpi.edu/artifactory/api/download/release/edu/wpi/first/$artifact"
  if [[ $artifact != wpinet && ! -f $cache/$artifact.jar ]]; then
    curl -fLsS "$base/$artifact-java/$version/$artifact-java-$version.jar" \
      -o "$cache/$artifact.jar"
  fi
  if [[ ! -f $cache/native/lib${artifact}jni.so ]]; then
    curl -fLsS "$base/$artifact-cpp/$version/$artifact-cpp-$version-linuxx86-64.zip" \
      -o "$cache/$artifact.zip"
    unzip -qo "$cache/$artifact.zip" -d "$cache/native"
    find "$cache/native" -name '*.so' -mindepth 2 -exec cp -t "$cache/native" {} +
  fi
done
java_bin=${JAVA_HOME:+$JAVA_HOME/bin/}
"${java_bin}javac" -Xlint:all -Werror -cp "$cache/ntcore.jar:$cache/wpiutil.jar" \
  -d "$cache/classes" "$test_dir/ClusterNT.java" "$test_dir/ClusterSubscriber.java"
LD_LIBRARY_PATH="$cache/native${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "${java_bin}java" -Djava.library.path="$cache/native" \
  -cp "$cache/classes:$cache/ntcore.jar:$cache/wpiutil.jar" ClusterSubscriber "$port"
