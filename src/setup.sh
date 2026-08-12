#!/usr/bin/env bash
# ELAJ environment setup.
# Creates a fresh conda environment, installs PyTorch with CUDA, and verifies
# the installation by running the test suite.
#
# Usage:
#   bash setup.sh          # default: CUDA 11.8
#   bash setup.sh 12.1     # CUDA 12.x
#   bash setup.sh cpu      # CPU-only (no GPU)

set -euo pipefail

CUDA_VERSION="${1:-11.8}"
ENV_NAME="elaj"
PYTHON_VERSION="3.10"

case "$CUDA_VERSION" in
    11.8) TORCH_INDEX="https://download.pytorch.org/whl/cu118" ;;
    12.1) TORCH_INDEX="https://download.pytorch.org/whl/cu121" ;;
    12.4) TORCH_INDEX="https://download.pytorch.org/whl/cu124" ;;
    cpu)  TORCH_INDEX="https://download.pytorch.org/whl/cpu" ;;
    *)
        echo "Unsupported CUDA version: $CUDA_VERSION"
        echo "Supported: 11.8 | 12.1 | 12.4 | cpu"
        exit 1
        ;;
esac

echo "========================================"
echo " ELAJ Setup"
echo " Conda env : $ENV_NAME"
echo " Python    : $PYTHON_VERSION"
echo " CUDA      : $CUDA_VERSION"
echo "========================================"

# 1. Create conda environment (fails gracefully if it already exists)
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[1/5] Conda env '$ENV_NAME' already exists — skipping creation."
else
    echo "[1/5] Creating conda env '$ENV_NAME' (Python $PYTHON_VERSION)..."
    conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
fi

# Activate conda in non-interactive shell
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# 2. Install PyTorch first (before the package, to avoid pulling in the CPU wheel)
echo "[2/5] Installing PyTorch (index: $TORCH_INDEX)..."
pip install torch --index-url "$TORCH_INDEX" --quiet

# 3. Install ELAJ with dev dependencies
echo "[3/5] Installing ELAJ package..."
pip install -e ".[dev]" --quiet

# 4. Verify installation
echo "[4/5] Verifying installation..."
python -c "
import torch, endo_model
print(f'  PyTorch {torch.__version__}')
print(f'  CUDA available : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU            : {torch.cuda.get_device_name(0)}')
print(f'  ELAJ {endo_model.__version__}')
"

# 5. Run test suite
echo "[5/5] Running tests..."
pytest tests/ -v

echo ""
echo "========================================"
echo " Setup complete."
echo " Activate with: conda activate $ENV_NAME"
echo "========================================"
