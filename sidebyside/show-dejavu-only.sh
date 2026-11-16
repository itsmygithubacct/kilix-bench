#!/bin/sh
# Print each character DejaVu has and Neuronte does not.
# Run it at the prompt of a side-by-side pane. The Neuronte panes draw these
# anyway, with a net that never saw them; F3 draws a plain mark there instead.
#   show-dejavu-only.sh [--against v2|n014|both]
here=$(cd "$(dirname "$0")" && pwd)
. "$here/env.sh"
python3 "$here/charsets.py" dejavu-only "$@"
python3 "$here/charsets.py" summary "$@" | head -6
