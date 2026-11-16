#!/bin/sh
# The three-way benchmark: Neuronte 2, Neuronte 0.1.4 and DejaVu drawing terminal cells.
#   bench.sh [--repeats N] [--cells 8x16,12x24] [--json FILE]
here=$(cd "$(dirname "$0")" && pwd)
. "$here/env.sh"
exec python3 "$here/bench.py" "$@"
