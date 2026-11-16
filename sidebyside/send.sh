#!/bin/sh
# Send text to every pane of the side-by-side test: its arguments, or standard input.
#   send.sh 'Hello, world'        send.sh < notes.txt        send.sh --clear
here=$(cd "$(dirname "$0")" && pwd)
stream=$(python3 -c 'import sys; sys.path.insert(0, sys.argv[1]); import pane; print(pane.stream_path())' "$here") || exit 1
case "$1" in
--clear) printf '\014' >> "$stream" ;;
--marks) printf '\025' >> "$stream" ;;
--size) printf '\024' >> "$stream" ;;
--up) printf '\002' >> "$stream" ;;
--down) printf '\006' >> "$stream" ;;
"") cat >> "$stream" ;;
*) printf '%s\n' "$*" >> "$stream" ;;
esac
