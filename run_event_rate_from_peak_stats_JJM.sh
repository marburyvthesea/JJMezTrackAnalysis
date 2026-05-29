#!/bin/bash
#SBATCH -A p30771
#SBATCH -p normal
#SBATCH --job-name=event-rate
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH -o ./logfiles/event-rate-%j.out

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  sbatch run_event_rate_from_peak_stats_JJM.sh /path/to/all_peak_stats.csv [--window-len N] [--output-csv /path/to/output.csv] [--gcamp-csv /path/to/GCAMP_with_velocity.csv] [--n-rows N] [--labels-mat /path/to/precomputed_output_LABELS.mat] [--label-source overall|ex|ml] [--conda-env ENV] [--repo-dir /path/to/JJMezTrackAnalysis]

Examples:
  sbatch run_event_rate_from_peak_stats_JJM.sh /scratch/jma819/CaliAli_linearTrackData/m388_analysis/all_peak_stats_20260518_211705.csv
  sbatch run_event_rate_from_peak_stats_JJM.sh /scratch/jma819/CaliAli_linearTrackData/m388_analysis/all_peak_stats_20260518_211705.csv --window-len 20
  sbatch run_event_rate_from_peak_stats_JJM.sh /scratch/jma819/CaliAli_linearTrackData/m388_analysis/all_peak_stats_20260518_211705.csv --labels-mat /scratch/jma819/CaliAli_linearTrackData/m388_analysis/precomputed_output_LABELS.mat --label-source overall
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

PEAK_STATS_CSV=""
WINDOW_LEN="20"
OUTPUT_CSV=""
GCAMP_CSV=""
N_ROWS=""
LABELS_MAT=""
LABEL_SOURCE="overall"
CONDA_ENV="jupyter-kernel-py38"
REPO_DIR="$HOME/JJMezTrackAnalysis"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --window-len)
            WINDOW_LEN="$2"
            shift 2
            ;;
        --output-csv)
            OUTPUT_CSV="$2"
            shift 2
            ;;
        --gcamp-csv)
            GCAMP_CSV="$2"
            shift 2
            ;;
        --n-rows)
            N_ROWS="$2"
            shift 2
            ;;
        --labels-mat)
            LABELS_MAT="$2"
            shift 2
            ;;
        --label-source)
            LABEL_SOURCE="$2"
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
            if [[ -z "$PEAK_STATS_CSV" ]]; then
                PEAK_STATS_CSV="$1"
            else
                echo "Unexpected positional argument: $1" >&2
                usage
                exit 1
            fi
            shift
            ;;
    esac
done

if [[ -z "$PEAK_STATS_CSV" ]]; then
    echo "Missing peak-stats CSV path" >&2
    usage
    exit 1
fi

if [[ ! -f "$PEAK_STATS_CSV" ]]; then
    echo "Input file not found: $PEAK_STATS_CSV" >&2
    exit 1
fi

SCRIPT_PATH="$REPO_DIR/scripts/compute_event_rate_from_peak_stats.py"
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
    --peak-stats-csv "$PEAK_STATS_CSV"
    --window-len "$WINDOW_LEN"
)

if [[ -n "$OUTPUT_CSV" ]]; then
    CMD+=(--output-csv "$OUTPUT_CSV")
fi
if [[ -n "$GCAMP_CSV" ]]; then
    CMD+=(--gcamp-csv "$GCAMP_CSV")
fi
if [[ -n "$N_ROWS" ]]; then
    CMD+=(--n-rows "$N_ROWS")
fi
if [[ -n "$LABELS_MAT" ]]; then
    CMD+=(--labels-mat "$LABELS_MAT" --label-source "$LABEL_SOURCE")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
