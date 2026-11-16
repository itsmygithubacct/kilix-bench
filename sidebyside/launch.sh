#!/bin/sh
# Open a Kilix tab with three panes side by side: Neuronte 2, Neuronte 0.1.4, DejaVu.
#   launch.sh [--cell 12x24]
# Type in any pane and the text appears in all three.
here=$(cd "$(dirname "$0")" && pwd)
. "$here/env.sh"
if [ "$1" = "--pane" ]; then
    # Inside the tab: open the next pane to the right, then become this pane.
    which=$2; shift 2
    case "$which" in
    v2) next=n014 ;;
    n014) next=dejavu ;;
    *) next= ;;
    esac
    [ -n "$next" ] && kilix pane right --title "sidebyside-$next" -- "$here/launch.sh" --pane "$next" "$@" >/dev/null
    exec python3 "$here/pane.py" "$which" "$@"
fi
command -v kilix >/dev/null || { echo "launch: kilix is not on PATH" >&2; exit 1; }
: > "$(python3 -c 'import sys; sys.path.insert(0, sys.argv[1]); import pane; print(pane.stream_path())' "$here")"
opened=$(kilix new-tab --title "Neuronte side by side" "$here/launch.sh" --pane v2 "$@") || exit 1
echo "$opened"
# The first pane is half the tab and the other two a quarter each: make them thirds.
first=${opened##* }
kitten=${KILIX_KITTEN:-$(command -v kitten)}
[ -n "$KITTY_LISTEN_ON" ] && [ -x "$kitten" ] || exit 0
for try in 1 2 3 4 5 6 7 8 9 10; do
    sleep 0.5
    narrow=$(kilix pane list --json 2>/dev/null | python3 -c '
import json, sys
def walk(o):
    if isinstance(o, dict):
        if o.get("title") == "sidebyside-dejavu":
            print(o.get("columns", 0))
        for v in o.values():
            walk(v)
    elif isinstance(o, list):
        for v in o:
            walk(v)
walk(json.load(sys.stdin))' | tail -1)
    [ -n "$narrow" ] && [ "$narrow" -gt 0 ] && break
done
[ -n "$narrow" ] && "$kitten" @ --to "$KITTY_LISTEN_ON" resize-window --match "id:$first" --axis horizontal \
    --increment "-$(( narrow * 5 / 8 ))" >/dev/null 2>&1
exit 0
