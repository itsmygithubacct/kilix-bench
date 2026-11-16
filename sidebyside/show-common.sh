#!/bin/sh
# Print each character Neuronte 2, Neuronte 0.1.4 and DejaVu all have.
# Run it at the prompt of a side-by-side pane: all three panes run it, and
# each draws the characters with its own face.
here=$(cd "$(dirname "$0")" && pwd)
. "$here/env.sh"
python3 "$here/charsets.py" common "$@"
python3 "$here/charsets.py" summary | head -6
