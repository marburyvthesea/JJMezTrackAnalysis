#!/usr/bin/env bash
set -euo pipefail

SRC_EP="8f796c9e-f5c8-11e5-9842-22000b9da45e"   # RDSS
DST_EP="d5990400-6d04-11e5-ba46-22000b92c6ec"   # Quest

SRC_BASE="/rdss/jma819/fsmresfiles/Basic_Sciences/Phys/ContractorLab/Projects/YZ/Miniscope_data/Miniscope_data/Linear_track"
DST_BASE="/scratch/jma819/behavCamData/YZ_linearTrackExperiments"

LABEL="behavcam_rotated_cropped_avi_to_quest_$(date +%Y%m%d_%H%M%S)"
BATCH="/tmp/${LABEL}.txt"
: > "$BATCH"

for mouse_dir in 311 326 328 388 752 757 992 994; do
  src_dir="${SRC_BASE}/BehavCamConcactenated_${mouse_dir}/rotated_and_cropped_avi"
  dst_dir="${DST_BASE}/${mouse_dir}"

  if ! globus ls "${SRC_EP}:${src_dir}/" >/dev/null 2>&1; then
    echo "Skipping missing source dir: ${src_dir}"
    continue
  fi

  globus mkdir "${DST_EP}:${dst_dir}" 2>/dev/null || true

  globus ls "${SRC_EP}:${src_dir}/" \
    | tr -d '\r' \
    | sed '/^$/d' \
    | grep -E '\.avi$' \
    | sort -V \
    | while IFS= read -r f; do
        printf "%s/%s %s/%s\n" "$src_dir" "$f" "$dst_dir" "$f" >> "$BATCH"
      done
done

nfiles="$(wc -l < "$BATCH" | tr -d ' ')"
if [[ "$nfiles" -eq 0 ]]; then
  echo "ERROR: No AVI files found to transfer."
  exit 1
fi

echo "Built batch file: $BATCH"
echo "Total files: $nfiles"

task_id=$(globus transfer "$SRC_EP" "$DST_EP" --label "$LABEL" --batch "$BATCH" | awk '/Task ID:/ {print $3}')
echo "Submitted Globus task: $task_id"
echo "Check status with:"
echo "  globus task show $task_id"
