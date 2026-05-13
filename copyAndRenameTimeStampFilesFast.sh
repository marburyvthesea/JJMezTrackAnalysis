#!/bin/bash

base="/Volumes/fsmresfiles/Basic_Sciences/Phys/ContractorLab/Projects/YZ/Miniscope_data/Miniscope_data/Linear_track"
mouse_num="311"

dest_webcam="$base/timeStampsBehavCam_copied_${mouse_num}"
dest_miniscope="$base/timeStampsMiniscope_copied_${mouse_num}"

mkdir -p "$dest_webcam" "$dest_miniscope"

cd "$base" || exit 1
shopt -s nullglob

count=0

for f in */${mouse_num}_*/My_WebCam/timeStamps.csv */${mouse_num}_*/My_V4_Miniscope/timeStamps.csv; do
    [[ -e "$f" ]] || continue
    count=$((count + 1))

    session=$(basename "$(dirname "$(dirname "$f")")")
    day=$(basename "$(dirname "$(dirname "$(dirname "$f")")")")
    parent=$(basename "$(dirname "$f")")

    if [[ "$parent" == "My_WebCam" ]]; then
        newname="${day}_${session}_timeStampsBehavCam.csv"
        dest="$dest_webcam/$newname"
    elif [[ "$parent" == "My_V4_Miniscope" ]]; then
        newname="${day}_${session}_timeStampsMiniscope.csv"
        dest="$dest_miniscope/$newname"
    else
        continue
    fi

    cp "$f" "$dest"
    echo "[$count] Copied: $f -> $dest"
done

echo "Done. Copied $count files."
