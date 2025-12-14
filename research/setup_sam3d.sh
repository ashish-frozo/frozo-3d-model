#!/bin/bash
# ============================================================================
# SAM-3D Objects Setup Script
# ============================================================================
# This script sets up the SAM-3D Objects environment for research benchmarking.
# 
# REQUIREMENTS:
# - Linux 64-bit OS
# - NVIDIA GPU with ≥32GB VRAM
# - Mamba/Conda installed
# - HuggingFace account with access to facebook/sam-3d-objects
#
# USAGE:
#   ./setup_sam3d.sh
#
# ⚠️  FOR INTERNAL RESEARCH USE ONLY - NOT FOR PRODUCTION
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}============================================${NC}"
echo -e "${YELLOW}  SAM-3D Objects Research Setup${NC}"
echo -e "${YELLOW}============================================${NC}"
echo ""
echo -e "${RED}⚠️  WARNING: For internal research only!${NC}"
echo -e "${RED}    Do NOT use outputs in production.${NC}"
echo ""

# ----------------------------------------------------------------------------
# Step 0: Check prerequisites
# ----------------------------------------------------------------------------
echo -e "${GREEN}[1/5] Checking prerequisites...${NC}"

# Check for Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo -e "${RED}Error: SAM-3D requires Linux. Detected: $OSTYPE${NC}"
    echo "You can still prepare the setup, but inference won't work on macOS."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for mamba/conda
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
elif command -v conda &> /dev/null; then
    CONDA_CMD="conda"
else
    echo -e "${RED}Error: mamba or conda not found. Please install miniconda/mamba.${NC}"
    exit 1
fi

echo "Using: $CONDA_CMD"

# Check for git
if ! command -v git &> /dev/null; then
    echo -e "${RED}Error: git not found.${NC}"
    exit 1
fi

# ----------------------------------------------------------------------------
# Step 1: Clone SAM-3D Objects repository
# ----------------------------------------------------------------------------
echo -e "${GREEN}[2/5] Cloning SAM-3D Objects repository...${NC}"

SAM3D_DIR="sam-3d-objects"

if [ -d "$SAM3D_DIR" ]; then
    echo "Directory $SAM3D_DIR already exists. Pulling latest..."
    cd "$SAM3D_DIR"
    git pull
    cd ..
else
    git clone https://github.com/facebookresearch/sam-3d-objects.git "$SAM3D_DIR"
fi

# ----------------------------------------------------------------------------
# Step 2: Create conda environment
# ----------------------------------------------------------------------------
echo -e "${GREEN}[3/5] Creating conda environment...${NC}"

cd "$SAM3D_DIR"

# Check if environment already exists
if $CONDA_CMD env list | grep -q "sam3d-objects"; then
    echo "Environment 'sam3d-objects' already exists. Skipping creation."
else
    $CONDA_CMD env create -f environments/default.yml
fi

# ----------------------------------------------------------------------------
# Step 3: Install dependencies
# ----------------------------------------------------------------------------
echo -e "${GREEN}[4/5] Installing dependencies...${NC}"

# Activate environment and install
cat << 'EOF' > install_deps.sh
#!/bin/bash
set -e

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate sam3d-objects

# Set pip indexes
export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"

# Install core package
pip install -e '.[dev]'
pip install -e '.[p3d]'

# Install inference dependencies
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
pip install -e '.[inference]'

# Apply patches
if [ -f "./patching/hydra" ]; then
    ./patching/hydra
fi

echo "Dependencies installed successfully!"
EOF

chmod +x install_deps.sh
echo "Run './sam-3d-objects/install_deps.sh' to complete installation."
echo "(Requires activating the sam3d-objects environment first)"

cd ..

# ----------------------------------------------------------------------------
# Step 4: Download checkpoints from HuggingFace
# ----------------------------------------------------------------------------
echo -e "${GREEN}[5/5] Preparing checkpoint download...${NC}"

cat << 'EOF' > download_checkpoints.sh
#!/bin/bash
# Download SAM-3D Objects checkpoints from HuggingFace
# 
# PREREQUISITES:
# 1. Request access at: https://huggingface.co/facebook/sam-3d-objects
# 2. Login with: huggingface-cli login

set -e

# Check for huggingface-cli
if ! command -v huggingface-cli &> /dev/null; then
    echo "Installing huggingface_hub..."
    pip install 'huggingface-hub[cli]<1.0'
fi

# Check if logged in
if ! huggingface-cli whoami &> /dev/null; then
    echo "Please login to HuggingFace:"
    huggingface-cli login
fi

# Download checkpoints
TAG=hf
CHECKPOINT_DIR="sam-3d-objects/checkpoints"

echo "Downloading checkpoints to $CHECKPOINT_DIR..."
huggingface-cli download \
    --repo-type model \
    --local-dir "${CHECKPOINT_DIR}/${TAG}-download" \
    --max-workers 1 \
    facebook/sam-3d-objects

# Move to final location
mv "${CHECKPOINT_DIR}/${TAG}-download/checkpoints" "${CHECKPOINT_DIR}/${TAG}"
rm -rf "${CHECKPOINT_DIR}/${TAG}-download"

echo "Checkpoints downloaded to: ${CHECKPOINT_DIR}/${TAG}"
EOF

chmod +x download_checkpoints.sh

# ----------------------------------------------------------------------------
# Create samples directory
# ----------------------------------------------------------------------------
mkdir -p samples
mkdir -p metrics

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  Setup Complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "Next steps:"
echo ""
echo "1. Complete SAM-3D installation:"
echo "   cd sam-3d-objects && ./install_deps.sh"
echo ""
echo "2. Download checkpoints (requires HuggingFace access):"
echo "   ./download_checkpoints.sh"
echo ""
echo "3. Add test images to: research/samples/"
echo ""
echo "4. Run benchmark:"
echo "   python benchmark.py --input_dir samples/"
echo ""
echo -e "${YELLOW}Remember: 32GB+ VRAM GPU required for inference!${NC}"
