#!/bin/bash

# RubAI Backend Deployment Script for Linux Server
# Run this script as root or with sudo

set -e

APP_DIR="/var/www/rubai-ragbot-backend"
APP_USER="www-data"
APP_GROUP="www-data"

echo "🚀 Starting RubAI Backend Deployment..."

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "❌ Please run as root or with sudo"
    exit 1
fi

# Create application directory
echo "📁 Creating application directory..."
mkdir -p $APP_DIR
chown $APP_USER:$APP_GROUP $APP_DIR

# Copy application files (assuming you're running this from the project root)
echo "📦 Copying application files..."
cp -r . $APP_DIR/
cd $APP_DIR

# Remove unnecessary files
rm -rf .git __pycache__ *.pyc .env venv

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Set proper permissions
echo "🔒 Setting permissions..."
chown -R $APP_USER:$APP_GROUP $APP_DIR
chmod -R 755 $APP_DIR
chmod 600 .env 2>/dev/null || echo "⚠️  .env file not found - you'll need to create it"

# Install systemd service
echo "⚙️  Installing systemd service..."
cp deploy/rubai-backend.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable rubai-backend

# Start service
echo "▶️  Starting service..."
systemctl start rubai-backend

# Check status
echo "📊 Service status:"
systemctl status rubai-backend --no-pager

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📝 Next steps:"
echo "1. Edit /var/www/rubai-ragbot-backend/.env with your configuration"
echo "2. Configure nginx (copy deploy/nginx.conf to /etc/nginx/sites-available/)"
echo "3. Restart the service: sudo systemctl restart rubai-backend"
echo "4. Check logs: sudo journalctl -u rubai-backend -f"

