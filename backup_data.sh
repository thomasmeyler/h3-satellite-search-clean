#!/bin/bash
# Script to backup large data files to a compressed archive

# Create backup directory if it doesn't exist
BACKUP_DIR="/workspace/h3_satellite_search_backup"
mkdir -p $BACKUP_DIR

# Backup raw data
echo "Backing up raw satellite imagery..."
tar -czf $BACKUP_DIR/raw_data.tar.gz -C /workspace/h3_satellite_search/data raw

# Backup embeddings
echo "Backing up embeddings..."
tar -czf $BACKUP_DIR/embeddings.tar.gz -C /workspace/h3_satellite_search/data embeddings

# Backup vector database
echo "Backing up vector database..."
tar -czf $BACKUP_DIR/vector_db.tar.gz -C /workspace/h3_satellite_search/data/processed vector_db

echo "Backup completed successfully!"
echo "Files are stored in $BACKUP_DIR"
echo ""
echo "To restore these files in a new Docker container:"
echo "1. Copy the backup files to the new container"
echo "2. Run the restore_data.sh script"
echo ""
echo "To copy files out of Docker to your local machine:"
echo "docker cp <container_id>:$BACKUP_DIR <local_destination>"
echo ""
echo "To copy files back into a new Docker container:"
echo "docker cp <local_source> <new_container_id>:$BACKUP_DIR"