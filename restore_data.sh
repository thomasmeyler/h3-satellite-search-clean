#!/bin/bash
# Script to restore large data files from a compressed archive

# Check if backup directory exists
BACKUP_DIR="/workspace/h3_satellite_search_backup"
if [ ! -d "$BACKUP_DIR" ]; then
    echo "Error: Backup directory $BACKUP_DIR not found!"
    echo "Please copy the backup files to this directory first."
    exit 1
fi

# Create data directories if they don't exist
mkdir -p /workspace/h3_satellite_search/data/raw
mkdir -p /workspace/h3_satellite_search/data/embeddings
mkdir -p /workspace/h3_satellite_search/data/processed/vector_db

# Restore raw data
if [ -f "$BACKUP_DIR/raw_data.tar.gz" ]; then
    echo "Restoring raw satellite imagery..."
    tar -xzf $BACKUP_DIR/raw_data.tar.gz -C /workspace/h3_satellite_search/data
else
    echo "Warning: Raw data backup not found at $BACKUP_DIR/raw_data.tar.gz"
fi

# Restore embeddings
if [ -f "$BACKUP_DIR/embeddings.tar.gz" ]; then
    echo "Restoring embeddings..."
    tar -xzf $BACKUP_DIR/embeddings.tar.gz -C /workspace/h3_satellite_search/data
else
    echo "Warning: Embeddings backup not found at $BACKUP_DIR/embeddings.tar.gz"
fi

# Restore vector database
if [ -f "$BACKUP_DIR/vector_db.tar.gz" ]; then
    echo "Restoring vector database..."
    tar -xzf $BACKUP_DIR/vector_db.tar.gz -C /workspace/h3_satellite_search/data/processed
else
    echo "Warning: Vector database backup not found at $BACKUP_DIR/vector_db.tar.gz"
fi

echo "Restoration completed!"
echo "You can now run the project with the restored data."