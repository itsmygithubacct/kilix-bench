#!/bin/sh
# Show, in every pane, each character Neuronte 2, Neuronte 0.1.4 and DejaVu all have.
here=$(cd "$(dirname "$0")" && pwd)
. "$here/env.sh"
"$here/send.sh" --clear
python3 "$here/charsets.py" common "$@" | "$here/send.sh"
python3 "$here/charsets.py" summary | head -6
echo "PgUp and PgDn in a pane scroll all three."
