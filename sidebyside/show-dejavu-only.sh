#!/bin/sh
# Show, in every pane, each character the DejaVu font has and Neuronte does not.
# The Neuronte panes draw them anyway, with a net that never saw them; Ctrl-U in
# a pane (or send.sh --marks) draws a plain mark there instead.
#   show-dejavu-only.sh [--against v2|n014|both]
here=$(cd "$(dirname "$0")" && pwd)
. "$here/env.sh"
"$here/send.sh" --clear
python3 "$here/charsets.py" dejavu-only "$@" | "$here/send.sh"
python3 "$here/charsets.py" summary "$@" | head -6
echo "PgUp and PgDn in a pane scroll all three."
