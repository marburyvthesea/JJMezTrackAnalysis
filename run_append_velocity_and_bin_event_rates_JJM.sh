#!/bin/bash
#SBATCH -A p30771
#SBATCH -p short
#SBATCH --job-name=event-bin
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH -o ./logfiles/event-bin-%j.out

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  sbatch run_append_velocity_and_bin_event_rates_JJM.sh /path/to/eventsPerSecond_mouse1.csv [/path/to/eventsPerSecond_mouse2.csv ...] [--base-dir /path/to/CaliAli_linearTrackData] [--window-len N] [--velocity-column COLUMN] [--per-mouse-output-name NAME.csv] [--summary-output-csv /path/to/summary.csv] [--conda-env ENV] [--repo-dir /path/to/JJMezTrackAnalysis]

Examples:
  sbatch run_append_velocity_and_bin_event_rates_JJM.sh /scratch/jma819/CaliAli_linearTrackData/m328_analysis/all_peak_stats_20260518_152949_events_per_second_window20_20260528_233031.csv /scratch/jma819/CaliAli_linearTrackData/m752_analysis/all_peak_stats_20260518_211705_events_per_second_window20_20260528_232058.csv --base-dir /scratch/jma819/CaliAli_linearTrackData
  sbatch run_append_velocity_and_bin_event_rates_JJM.sh /scratch/jma819/CaliAli_linearTrackData/m311_analysis/eventsPerSecond_df.csv /scratch/jma819/CaliAli_linearTrackData/m326_analysis/eventsPerSecond_df.csv --base-dir /scratch/jma819/CaliAli_linearTrackData --window-len 20
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

EVENTS_CSVS=()
BASE_DIR="."
WINDOW_LEN="20"
VELOCITY_COLUMN="Velocity_spatial_filtered"
PER_MOUSE_OUTPUT_NAME=""
SUMMARY_OUTPUT_CSV=""
CONDA_ENV="jupyter-kernel-py38"
REPO_DIR="$HOME/JJMezTrackAnalysis"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base-dir)
            BASE_DIR="$2"
            shift 2
            ;;
        --window-len)
            WINDOW_LEN="$2"
            shift 2
            ;;
        --velocity-column)
            VELOCITY_COLUMN="$2"
            shift 2
            ;;
        --per-mouse-output-name)
            PER_MOUSE_OUTPUT_NAME="$2"
            shift 2
            ;;
        --summary-output-csv)
            SUMMARY_OUTPUT_CSV="$2"
            shift 2
            ;;
        --conda-env)
            CONDA_ENV="$2"
            shift 2
            ;;
        --repo-dir)
            REPO_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
        *)
            EVENTS_CSVS+=("$1")
            shift
            ;;
    esac
done

if [[ ${#EVENTS_CSVS[@]} -eq 0 ]]; then
    echo "At least one event-rate CSV is required." >&2
    usage
    exit 1
fi

SCRIPT_PATH="$REPO_DIR/scripts/append_velocity_and_bin_event_rates.py"
if [[ ! -f "$SCRIPT_PATH" ]]; then
    echo "Python script not found: $SCRIPT_PATH" >&2
    exit 1
fi

if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    # shellcheck source=/dev/null
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
else
    echo "conda not found in PATH" >&2
    exit 1
fi

export PYTHONUNBUFFERED=1

CMD=(
    python -u "$SCRIPT_PATH"
)

for csv_path in "${EVENTS_CSVS[@]}"; do
    CMD+=("$csv_path")
done

CMD+=(
    --base-dir "$BASE_DIR"
    --window-len "$WINDOW_LEN"
    --velocity-column "$VELOCITY_COLUMN"
)

if [[ -n "$PER_MOUSE_OUTPUT_NAME" ]]; then
    CMD+=(--per-mouse-output-name "$PER_MOUSE_OUTPUT_NAME")
fi
if [[ -n "$SUMMARY_OUTPUT_CSV" ]]; then
    CMD+=(--summary-output-csv "$SUMMARY_OUTPUT_CSV")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
