base="/Volumes/fsmresfiles/Basic_Sciences/Phys/ContractorLab/Projects/YZ/Miniscope_data/Miniscope_data/Linear_track"
dest="$base/timeStampsBehavCam_copied"

mkdir -p "$dest"

find "$base" -type f -path '*/328_*/My_WebCam/timeStamps.csv' -print0 |
while IFS= read -r -d '' f; do
    session=$(basename "$(dirname "$(dirname "$f")")")
    day=$(basename "$(dirname "$(dirname "$(dirname "$f")")")")
    newname="${day}_${session}_timeStampsBehavCam.csv"
    cp "$f" "$dest/$newname"
    echo "Copied: $f -> $dest/$newname"
done