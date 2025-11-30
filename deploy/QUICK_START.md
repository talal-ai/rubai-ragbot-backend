# Quick Start - Linux Server Deployment

## Fast Deployment Steps

### 1. Upload Files to Server

```bash
# From your local machine, upload to server
scp -r . user@your-server-ip:/tmp/rubai-backend
```

### 2. On Your Server - Run Deployment

```bash
# SSH into your server
ssh user@your-server-ip

# Move to /var/www
sudo mv /tmp/rubai-backend /var/www/rubai-ragbot-backend

# Run deployment script
cd /var/www/rubai-ragbot-backend
sudo chmod +x deploy/deploy.sh
sudo ./deploy/deploy.sh
```

### 3. Configure Environment

```bash
# Edit .env file with your settings
sudo nano /var/www/rubai-ragbot-backend/.env
```

### 4. Restart Service

```bash
sudo systemctl restart rubai-backend
sudo systemctl status rubai-backend
```

### 5. Configure Firewall (Allow Port 5555)

```bash
# Allow port 5555
sudo ufw allow 5555/tcp
sudo ufw enable
```

### 6. Test It

```bash
# Check if running
curl http://localhost:5555/docs

# Or from browser
http://your-server-ip:5555/docs
```

**Note:** Nginx is NOT required! Your app runs directly on port 5555.

## Manual Setup (Alternative)

If you prefer manual setup:

```bash
# 1. Create directory
sudo mkdir -p /var/www/rubai-ragbot-backend
sudo chown www-data:www-data /var/www/rubai-ragbot-backend

# 2. Copy files
sudo cp -r * /var/www/rubai-ragbot-backend/

# 3. Create venv
cd /var/www/rubai-ragbot-backend
sudo -u www-data python3 -m venv venv
sudo -u www-data venv/bin/pip install -r requirements.txt

# 4. Create .env
sudo -u www-data cp .env.example .env
sudo -u www-data nano .env

# 5. Install service
sudo cp deploy/rubai-backend.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable rubai-backend
sudo systemctl start rubai-backend
```

## Common Commands

```bash
# View logs
sudo journalctl -u rubai-backend -f

# Restart
sudo systemctl restart rubai-backend

# Stop
sudo systemctl stop rubai-backend

# Status
sudo systemctl status rubai-backend
```

