#!/bin/bash

# Update script for RubAI Backend
# Run this when you want to update your deployed application

set -e

APP_DIR="/var/www/rubai-ragbot-backend"
APP_USER="www-data"

echo "🔄 Updating RubAI Backend..."

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "❌ Please run as root or with sudo"
    exit 1
fi

cd $APP_DIR

# Backup .env file
echo "💾 Backing up .env..."
cp .env .env.backup 2>/dev/null || echo "⚠️  No .env file to backup"

# Pull latest changes (if using git)
if [ -d .git ]; then
    echo "📥 Pulling latest changes..."
    sudo -u $APP_USER git pull
else
    echo "⚠️  Not a git repository - skipping git pull"
fi

# Activate virtual environment and update dependencies
echo "📦 Updating dependencies..."
sudo -u $APP_USER $APP_DIR/venv/bin/pip install --upgrade pip
sudo -u $APP_USER $APP_DIR/venv/bin/pip install -r requirements.txt

# Restart service
echo "🔄 Restarting service..."
systemctl restart rubai-backend

# Wait a moment and check status
sleep 2
systemctl status rubai-backend --no-pager

echo ""
echo "✅ Update complete!"

