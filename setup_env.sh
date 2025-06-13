#!/usr/bin/env bash
set -euo pipefail

# Function to handle errors
error_exit() {
    echo "ERROR: $1" >&2
    exit 1
}

# Check if conda is available
if ! command -v conda &> /dev/null; then
    error_exit "'conda' command not found. Please ensure conda is installed and in your PATH."
fi

# Initialize conda for the current shell
eval "$(conda shell.bash hook 2>/dev/null)" || {
    echo "Warning: Could not initialize conda. Trying to source conda.sh..."
    # Try to find and source conda.sh
    if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    else
        error_exit "Could not initialize conda. Please ensure conda is properly installed."
    fi
}

# Configuration
DEV=0
ENV_NAME="dev_env"  # Default to dev environment

# Parse arguments
for arg in "$@"; do
    case "$arg" in
        --dev)
            DEV=1
            ENV_NAME="dev_env"
            shift
            ;;
        --prod)
            DEV=0
            ENV_NAME="prod_env"
            shift
            ;;
        *)
            echo "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

# Get absolute path to the environment directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PATH="${SCRIPT_DIR}/${ENV_NAME}"

# Create or update the environment
if [ -d "$ENV_PATH" ]; then
    echo "Updating conda environment in '${ENV_PATH}'..."
    conda env update --prefix "$ENV_PATH" --file "${SCRIPT_DIR}/environment.yml" --prune || {
        error_exit "Failed to update conda environment"
    }
else
    echo "Creating new conda environment in '${ENV_PATH}'..."
    conda env create --prefix "$ENV_PATH" --file "${SCRIPT_DIR}/environment.yml" || {
        error_exit "Failed to create conda environment"
    }
fi

# Install in development mode if --dev is specified
if [[ $DEV -eq 1 ]]; then
    echo "Installing in development mode..."
    conda run --no-capture-output --prefix "$ENV_PATH" pip install -e "${SCRIPT_DIR}[dev]" || {
        error_exit "Failed to install in development mode"
    }
fi

# Create activation script
cat > "${SCRIPT_DIR}/activate_env.sh" << 'EOL'
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
EOL
chmod +x "${SCRIPT_DIR}/activate_env.sh"

# Create .gitignore if it doesn't exist
if [ ! -f "${SCRIPT_DIR}/.gitignore" ]; then
    cat > "${SCRIPT_DIR}/.gitignore" << 'EOL'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Conda environments
dev_env/
prod_env/

# IDE
.idea/
.vscode/
*.swp
*.swo
.DS_Store

# Logs
logs/
*.log

# Environment files
.env
.venv
env/
venv/
ENV/

# Jupyter Notebook
.ipynb_checkpoints

# Local development
.pytest_cache/
.coverage
htmlcov/

# Distribution / packaging
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
EOL
    echo "Created .gitignore file"
fi

echo -e "\n✅ Environment setup complete!"
echo -e "To activate this environment, run:"
echo -e "    source \"${SCRIPT_DIR}/activate_env.sh\"\n"
