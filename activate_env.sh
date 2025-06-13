#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Initialize conda if not already done
if [ -z "$CONDA_SHLVL" ]; then
    if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    else
        eval "$(conda shell.bash hook 2>/dev/null)" || {
            echo "Error: Could not initialize conda"
            exit 1
        }
    fi
fi

# Try to activate the environment
if [ -d "${SCRIPT_DIR}/dev_env" ]; then
    conda deactivate 2>/dev/null || true
    conda activate "${SCRIPT_DIR}/dev_env"
elif [ -d "${SCRIPT_DIR}/prod_env" ]; then
    conda deactivate 2>/dev/null || true
    conda activate "${SCRIPT_DIR}/prod_env"
else
    echo "Error: No environment found. Run ./setup_env.sh [--dev|--prod] first."
    exit 1
fi
