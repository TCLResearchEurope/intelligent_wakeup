#!/bin/bash
# This code was developed by TCL Research Europe.
# Script for archiving generated text corpora to GitLab artifacts and NAS.

set -e  # Exit on error

# Configuration
# Optional shared-storage archive destination. No default is committed: set
# NAS_BASE_PATH to enable the copy, otherwise the archive is kept locally only.
NAS_BASE_PATH=${NAS_BASE_PATH:-""}
OUTPUT_DIR="dataset/data/generated_text"
ARCHIVE_DIR="artifacts"
DATE_STAMP=$(date -u +%Y-%m-%d)
COMMIT_HASH=${CI_COMMIT_SHORT_SHA:-$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")}

echo "=========================================="
echo "Text Corpora Archiving"
echo "=========================================="
echo "Date: $DATE_STAMP"
echo "Commit: $COMMIT_HASH"
echo "NAS Path: $NAS_BASE_PATH"
echo "=========================================="

# Create local artifact directory
mkdir -p "$ARCHIVE_DIR"

# Define archive filename
ARCHIVE_NAME="text-corpora-${DATE_STAMP}-${COMMIT_HASH}.tar.gz"
ARCHIVE_PATH="$ARCHIVE_DIR/$ARCHIVE_NAME"

# Create tar.gz archive
echo ""
echo "Creating archive: $ARCHIVE_NAME"
echo "Including:"
echo "  - Generated text files"
echo "  - Validation reports"
echo "  - Generation metadata"

tar -czf "$ARCHIVE_PATH" \
    -C dataset/data \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    generated_text/

# Verify archive was created
if [ -f "$ARCHIVE_PATH" ]; then
    ARCHIVE_SIZE=$(du -h "$ARCHIVE_PATH" | cut -f1)
    echo "✓ Archive created successfully: $ARCHIVE_PATH ($ARCHIVE_SIZE)"
else
    echo "✗ Failed to create archive"
    exit 1
fi

# Create manifest file for this archive
MANIFEST_PATH="$ARCHIVE_DIR/manifest-${DATE_STAMP}-${COMMIT_HASH}.json"
if [ -f "$OUTPUT_DIR/generation_metadata.json" ]; then
    # Merge generation metadata with archive info
    python3 -c "
import json
import os

# Load generation metadata
with open('$OUTPUT_DIR/generation_metadata.json', 'r') as f:
    metadata = json.load(f)

# Load validation manifest if available
validation_manifest = {}
if os.path.exists('$OUTPUT_DIR/validation/manifest.json'):
    with open('$OUTPUT_DIR/validation/manifest.json', 'r') as f:
        validation_manifest = json.load(f)

# Create combined manifest
manifest = {
    'archive_name': '$ARCHIVE_NAME',
    'archive_size_bytes': os.path.getsize('$ARCHIVE_PATH'),
    'date_stamp': '$DATE_STAMP',
    'commit_hash': '$COMMIT_HASH',
    'generation': metadata,
    'validation': validation_manifest,
}

# Save manifest
with open('$MANIFEST_PATH', 'w') as f:
    json.dump(manifest, f, indent=2)

print(f'Manifest created: $MANIFEST_PATH')
"
else
    # Create basic manifest
    cat > "$MANIFEST_PATH" <<EOF
{
  "archive_name": "$ARCHIVE_NAME",
  "date_stamp": "$DATE_STAMP",
  "commit_hash": "$COMMIT_HASH",
  "git_branch": "${CI_COMMIT_BRANCH:-unknown}",
  "pipeline_id": "${CI_PIPELINE_ID:-unknown}"
}
EOF
fi

echo "✓ Manifest created: $MANIFEST_PATH"

# Copy to NAS if path exists and is writable
if [ -d "$NAS_BASE_PATH" ] && [ -w "$NAS_BASE_PATH" ]; then
    echo ""
    echo "Copying to NAS..."

    # Create date-based directory on NAS
    NAS_DATE_DIR="$NAS_BASE_PATH/$DATE_STAMP"
    mkdir -p "$NAS_DATE_DIR"

    # Copy archive and manifest to NAS
    cp "$ARCHIVE_PATH" "$NAS_DATE_DIR/"
    cp "$MANIFEST_PATH" "$NAS_DATE_DIR/"

    # Also copy validation report if available
    if [ -f "$OUTPUT_DIR/validation/validation_report.html" ]; then
        cp "$OUTPUT_DIR/validation/validation_report.html" "$NAS_DATE_DIR/"
        echo "✓ Copied validation report to NAS"
    fi

    # Update 'latest' symlink
    LATEST_LINK="$NAS_BASE_PATH/latest"
    if [ -L "$LATEST_LINK" ]; then
        rm "$LATEST_LINK"
    fi
    ln -s "$NAS_DATE_DIR" "$LATEST_LINK"

    echo "✓ Copied to NAS: $NAS_DATE_DIR"
    echo "✓ Updated 'latest' symlink"

    # List NAS contents
    echo ""
    echo "NAS directory structure:"
    ls -lh "$NAS_DATE_DIR"

else
    echo ""
    if [ -z "$NAS_BASE_PATH" ]; then
        echo "NAS_BASE_PATH not set - archive saved locally only."
    else
        echo "⚠ NAS path not accessible: $NAS_BASE_PATH"
        echo "  Archive saved locally only."
    fi
fi

# Print summary
echo ""
echo "=========================================="
echo "Archiving Summary"
echo "=========================================="
echo "Archive: $ARCHIVE_NAME"
echo "Size: $(du -h "$ARCHIVE_PATH" | cut -f1)"
echo "GitLab Artifact: ✓ Available for download"

if [ -d "$NAS_BASE_PATH" ] && [ -w "$NAS_BASE_PATH" ]; then
    echo "NAS Copy: ✓ $NAS_BASE_PATH/$DATE_STAMP"
else
    echo "NAS Copy: ✗ Not available"
fi

echo "=========================================="
echo ""
echo "Archiving completed successfully."
exit 0
