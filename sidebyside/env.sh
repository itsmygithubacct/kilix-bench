# Where the two Neuronte trees and the DejaVu font are. Sourced by the scripts.
# Put the three settings in ~/.config/kilix-bench/sidebyside.env, or export them:
#   NEURONTE_V2_HOME=/path/to/neuronte-v2      built with make
#   NEURONTE_014_HOME=/path/to/neuronte        at v0.1.4, built with make
#   DEJAVU_FONT=Mono.ttf:Sans.ttf              optional; DejaVu is found in the usual places
#   NEURONTE_014_BUNDLE=neuronte-split-v5.nrt  optional; 0.1.4's default terminal font
conf=${XDG_CONFIG_HOME:-$HOME/.config}/kilix-bench/sidebyside.env
if [ -r "$conf" ]; then
    set -a
    . "$conf"
    set +a
fi
