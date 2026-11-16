#!/bin/sh
# Type into every pane of the side-by-side test from outside it: the arguments
# as one line, or standard input as it is.
#   send.sh './show-common.sh'      send.sh 'echo ✓ ≤ ¼'      send.sh --size
here=$(cd "$(dirname "$0")" && pwd)
stream=$(python3 -c 'import sys; sys.path.insert(0, sys.argv[1]); import pane; print(pane.stream_path())' "$here") || exit 1
case "$1" in
--up) printf '\034' >> "$stream" ;;
--size) printf '\035' >> "$stream" ;;
--marks) printf '\036' >> "$stream" ;;
--clear) printf '\037' >> "$stream" ;;
--down) printf '\000' >> "$stream" ;;
"") cat >> "$stream" ;;
*) printf '%s\n' "$*" >> "$stream" ;;
esac
