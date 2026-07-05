#!/usr/bin/env bash

# Source the manifest from this script's own dir (install-dir-name agnostic).
. "$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]:-$0}")")" && pwd)/h-manifest.conf"

conf=""
# Pool URL vacío en el flight sheet → omite -s (el binario usa su pool por defecto).
# Sin este guard, "-s --mining-address" hacía que clap tragase el flag como valor de -s.
[[ -n $CUSTOM_URL ]] && conf+=" -s $CUSTOM_URL"
conf+=" --mining-address $CUSTOM_TEMPLATE"

[[ ! -z $CUSTOM_USER_CONFIG ]] && conf+=" $CUSTOM_USER_CONFIG"

echo "$conf"
echo "$conf" > $CUSTOM_CONFIG_FILENAME
