#!/bin/bash
#SBATCH -A p30771
#SBATCH -p short
#SBATCH --job-name=peak-stats
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH -o ./logfiles/peak-stats-%j.out

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  sbatch run_peak_stats_from_gcamp_csv.sbatch /path/to/GCAMP_with_velocity.csv [--output-dir /path/to/output] [--conda-env ENV] [--repo-dir /path/to/JJMezTrackAnalysis] [-- extra python args]

Examples:
  sbatch slurm/run_peak_stats_from_gcamp_csv.sbatch /projects/b1118/CaliAli_linearTrackData/m328_analysis/GCAMP_with_velocity.csv
  sbatch slurm/run_peak_stats_from_gcamp_csv.sbatch /projects/b1118/CaliAli_linearTrackData/m328_analysis/GCAMP_with_velocity.csv --output-dir /projects/b1118/CaliAli_linearTrackData/m328_analysis
  sbatch slurm/run_peak_stats_from_gcamp_csv.sbatch /projects/b1118/CaliAli_linearTrackData/m328_analysis/GCAMP_with_velocity.csv -- --threshold 3.0 --min-peak-prominence 2.0
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

GCAMP_CSV=""
OUTPUT_DIR=""
CONDA_ENV="jupyter-kernel-py38"
REPO_DIR="$HOME/JJMezTrackAnalysis"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)
            OUTPUT_DIR="$2"
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
        --)
            shift
            EXTRA_ARGS=("$@")
            break
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
        *)
            if [[ -z "$GCAMP_CSV" ]]; then
                GCAMP_CSV="$1"
            else
                echo "Unexpected positional argument: $1" >&2
                usage
                exit 1
            fi
            shift
            ;;
    esac
done

if [[ -z "$GCAMP_CSV" ]]; then
    echo "Missing GCAMP_with_velocity.csv path" >&2
    usage
    exit 1
fi

if [[ ! -f "$GCAMP_CSV" ]]; then
    echo "Input file not found: $GCAMP_CSV" >&2
    exit 1
fi

SCRIPT_PATH="$REPO_DIR/scripts/compute_peak_stats_from_gcamp_csv.py"
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

CMD=(python "$SCRIPT_PATH" --gcamp-csv "$GCAMP_CSV")
if [[ -n "$OUTPUT_DIR" ]]; then
    CMD+=(--output-dir "$OUTPUT_DIR")
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_ARGS[@]}")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
