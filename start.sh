#!/usr/bin/env bash
# start.sh - Entrypoint script for Render deployment

# Ensure the /data directory exists (mounted persistent disk)
mkdir -p /data/uploads
mkdir -p /data/chroma_db

# Create symlinks from the /app directory to the /data persistent disk
# This ensures SQLite and ChromaDB survive deployments and restarts
ln -sfn /data/documents.db /app/documents.db
ln -sfn /data/chroma_db /app/chroma_db
ln -sfn /data/uploads /app/uploads

# Run honcho to start both the web and worker processes
exec honcho start
