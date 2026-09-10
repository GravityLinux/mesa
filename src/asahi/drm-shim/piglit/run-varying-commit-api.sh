#!/bin/bash
# SPDX-License-Identifier: MIT
# Run after a fresh m1n1 build/reset/chainload, just like run-graphics.sh.
set -euo pipefail
if [ "$#" != 1 ]; then
    echo "usage: $0 NEW_RESULTS_DIRECTORY" >&2
    exit 2
fi
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
mkdir -p -- "$1"
results=$(cd -- "$1" && pwd)
if [ -e "$results/run.log" ]; then
    echo "Refusing to replace $results/run.log" >&2
    exit 2
fi
${CC:-cc} -std=c11 -Wall -Wextra -Werror \
    "$script_dir/tests-graphics/varying-commit/api.c" -o "$results/varying-commit-api" \
    $(pkg-config --cflags --libs epoxy egl)
export ASAHI_GRAPHICS_BINARY="$results/varying-commit-api"
export ASAHI_GRAPHICS_EXPECTED_RESULTS=6
export T8132_PIGLIT_TIMEOUT=${T8132_PIGLIT_TIMEOUT:-900}
exec "$script_dir/run-graphics.sh" "$results"
