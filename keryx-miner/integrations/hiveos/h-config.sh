#!/usr/bin/env bash

# Source the manifest from this script's own dir (install-dir-name agnostic).
. "$(cd "$(dirname "$(readlink -f "$0")")" && pwd)/h-manifest.conf"

conf=""
conf+=" -s $CUSTOM_URL --mining-address $CUSTOM_TEMPLATE"

[[ ! -z $CUSTOM_USER_CONFIG ]] && conf+=" $CUSTOM_USER_CONFIG"

echo "$conf"
echo "$conf" > $CUSTOM_CONFIG_FILENAME
