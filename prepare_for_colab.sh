#!/bin/bash
# prepare_for_colab.sh
# This script creates a ZIP file with essential project files for Google Colab

echo "📦 Preparing project for Google Colab..."

# Create temporary directory
TEMP_DIR="frozo-3d-model-colab"
mkdir -p "$TEMP_DIR"

# Copy essential files
echo "Copying project files..."
cp -r svscn "$TEMP_DIR/"
cp -r scripts "$TEMP_DIR/"
cp requirements.txt "$TEMP_DIR/"
cp -r notebooks "$TEMP_DIR/" 2>/dev/null || true
cp README.md "$TEMP_DIR/" 2>/dev/null || true

# Create ZIP file
ZIP_NAME="frozo-3d-model-colab.zip"
echo "Creating ZIP file: $ZIP_NAME"
zip -r "$ZIP_NAME" "$TEMP_DIR" -x "*.pyc" -x "*__pycache__*" -x "*.git*"

# Cleanup
rm -rf "$TEMP_DIR"

echo ""
echo "✅ Done! Upload this file to Colab:"
echo "   $ZIP_NAME"
echo ""
echo "📋 In Colab, run:"
echo "   !unzip frozo-3d-model-colab.zip"
echo "   %cd frozo-3d-model-colab"
echo ""
echo "Then proceed with the notebook steps!"
