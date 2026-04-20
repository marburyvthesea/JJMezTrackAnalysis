base="/Volumes/fsmresfiles/Basic_Sciences/Phys/ContractorLab/Projects/YZ/Miniscope_data/Miniscope_data/Linear_track"
mouse_num="328"

dest_webcam="$base/timeStampsBehavCam_copied_${mouse_num}"
dest_miniscope="$base/timeStampsMiniscope_copied_${mouse_num}"

mkdir -p "$dest_webcam" "$dest_miniscope"

find "$base" -type f \( \
    -path "*/${mouse_num}_*/My_WebCam/timeStamps.csv" -o \
    -path "*/${mouse_num}_*/My_V4_Miniscope/timeStamps.csv" \
\) -print0 |
while IFS= read -r -d '' f; do
    session=$(basename "$(dirname "$(dirname "$f")")")
    day=$(basename "$(dirname "$(dirname "$(dirname "$f")")")")
    parent=$(basename "$(dirname "$f")")

    if [[ "$parent" == "My_WebCam" ]]; then
        newname="${day}_${session}_timeStampsBehavCam.csv"
        cp "$f" "$dest_webcam/$newname"
        echo "Copied: $f -> $dest_webcam/$newname"
    elif [[ "$parent" == "My_V4_Miniscope" ]]; then
        newname="${day}_${session}_timeStampsMiniscope.csv"
        cp "$f" "$dest_miniscope/$newname"
        echo "Copied: $f -> $dest_miniscope/$newname"
    fi
done