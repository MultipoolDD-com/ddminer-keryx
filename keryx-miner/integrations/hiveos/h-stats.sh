#!/usr/bin/env bash

# Force C locale: on comma-decimal locales awk parses "30.30" as 30, corrupting the kH/s scaling.
export LC_ALL=C

# Source the manifest from THIS script's own dir, so it works regardless of the install dir name
# HiveOS uses (keryx-miner vs keryx-miner-0.5.2, etc.).
. "$(cd "$(dirname "$(readlink -f "$0")")" && pwd)/h-manifest.conf"

LOG="$CUSTOM_LOG_BASENAME.log"

# env_logger line format: "[2026-07-01T09:35:44Z INFO  keryx_miner::miner] Current hashrate is 60.61 Mhash/s"
stats_raw=`grep "Current hashrate is" "$LOG" 2>/dev/null | tail -n 1`

maxDelay=120
time_now=`date +%s`

# Freshness: extract the ISO-8601 timestamp anywhere on the line (robust to env_logger format).
ts=`echo "$stats_raw" | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}' | head -1`
time_rep=`date -u -d "${ts/T/ }" +%s 2>/dev/null || echo 0`
diffTime=`echo $((time_now-time_rep)) | tr -d '-'`

# Convert "<value> <unit>hash/s" → khs (HiveOS reports in kH/s). value keeps 2 decimals → scale.
to_khs() {
	local line="$1"
	# second-to-last field = number, last field = unit (e.g. "60.61 Mhash/s")
	local v=`echo "$line" | awk '{print $(NF-1)}'`
	local u=`echo "$line" | awk '{print $NF}'`
	# value*100 as integer (avoids floats), then scale by unit to kH/s, then /100
	local vi=`echo "$v" | awk '{printf "%d", $1*100}'`
	local mul=1
	case "$u" in
		*Thash*) mul=1000000000 ;;   # TH → kH
		*Ghash*) mul=1000000 ;;      # GH → kH
		*Mhash*) mul=1000 ;;         # MH → kH
		*Khash*) mul=1 ;;            # kH → kH
		*hash/s) mul=0 ;;            # H/s → ~0 kH (rounds down)
	esac
	echo $(( vi*mul/100 ))
}

if [ "$diffTime" -lt "$maxDelay" ] && [ -n "$stats_raw" ]; then
	total_khs=`to_khs "$stats_raw"`

	# GPU status from HiveOS (busids/brand/temp/fan), aligned to the miner's mining GPUs only.
	readarray -t gpu_stats < <( jq --slurp -r -c '.[] | .busids, .brand, .temp, .fan | join(" ")' $GPU_STATS_JSON 2>/dev/null)
	busids=(${gpu_stats[0]})
	brands=(${gpu_stats[1]})
	temps=(${gpu_stats[2]})
	fans=(${gpu_stats[3]})
	gpu_count=${#busids[@]}

	hash_arr=(); busid_arr=(); fan_arr=(); temp_arr=()

	if [ $(gpu-detect NVIDIA) -gt 0 ]; then
		BRAND_MINER="nvidia"
	elif [ $(gpu-detect AMD) -gt 0 ]; then
		BRAND_MINER="amd"
	fi

	# The miner numbers its GPUs 0..N over MINING cards only (no iGPU). Keep a separate mining
	# index `m` so an iGPU in the HiveOS list doesn't shift "Device #N" off by one.
	m=0
	for(( i=0; i < gpu_count; i++ )); do
		[[ "${brands[i]}" != "$BRAND_MINER" ]] && continue
		[[ "${busids[i]}" =~ ^([A-Fa-f0-9]+): ]]
		busid_arr+=($((16#${BASH_REMATCH[1]})))
		temp_arr+=(${temps[i]})
		fan_arr+=(${fans[i]})
		gpu_raw=`grep "Device #$m:" "$LOG" 2>/dev/null | tail -n 1`
		hash_arr+=(`to_khs "$gpu_raw"`)
		m=$((m+1))
	done

	hash_json=`printf '%s\n' "${hash_arr[@]}" | jq -cs '.'`
	bus_numbers=`printf '%s\n' "${busid_arr[@]}" | jq -cs '.'`
	fan_json=`printf '%s\n' "${fan_arr[@]}" | jq -cs '.'`
	temp_json=`printf '%s\n' "${temp_arr[@]}" | jq -cs '.'`

	# Accepted / rejected shares for the `ar` field.
	shares_raw=`grep "Shares total:" "$LOG" 2>/dev/null | tail -n 1`
	acc=`echo "$shares_raw" | grep -oE '[0-9]+ accepted' | grep -oE '[0-9]+'`
	rej=`echo "$shares_raw" | grep -oE '[0-9]+ rejected' | grep -oE '[0-9]+'`
	[[ -z $acc ]] && acc=0
	[[ -z $rej ]] && rej=0

	uptime=$(( `date +%s` - `stat -c %Y $CUSTOM_CONFIG_FILENAME 2>/dev/null || echo $time_now` ))

	stats=$(jq -nc \
		--argjson hs "$hash_json" \
		--arg ver "$CUSTOM_VERSION" \
		--argjson bus_numbers "$bus_numbers" \
		--argjson fan "$fan_json" \
		--argjson temp "$temp_json" \
		--argjson acc "$acc" \
		--argjson rej "$rej" \
		--arg uptime "$uptime" \
		'{ hs: $hs, hs_units: "khs", algo: "keryxhash", ver: $ver, ar: [$acc, $rej], $uptime, $bus_numbers, $temp, $fan }')
	khs=$total_khs
else
	khs=0
	stats="null"
fi

echo "Log file : $LOG"
echo "Time since last log entry : $diffTime"
echo "Raw stats : $stats_raw"
echo "KHS : $khs"
echo "Output : $stats"

[[ -z $khs ]] && khs=0
[[ -z $stats ]] && stats="null"
