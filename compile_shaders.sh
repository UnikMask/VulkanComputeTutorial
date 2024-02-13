#!/bin/env sh
cd "$(dirname "$(realpath -- "$0")")";

for f in $(ls shaders); do
    if [ "${f##*.}" != "spv" ]; then
        glslc "shaders/$(basename -- $f)" -o "shaders/$f.spv"
    fi
done
