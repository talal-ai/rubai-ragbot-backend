# Linux Server Deployment Guide

This guide will help you deploy your RubAI Backend to a Linux server in `/var/www`.

## Prerequisites

- Linux server (Ubuntu/Debian recommended)
- Python 3.8 or higher
- PostgreSQL (or use Supabase)
- Nginx (optional - only if you want reverse proxy/SSL)
- Root or sudo access

**Note:** Nginx is completely optional! Your FastAPI app can run directly without it.

## Step 1: Prepare Your Server

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install python3 python3-pip python3-venv -y

# Install PostgreSQL (if not using Supabase)
sudo apt install postgresql postgresql-contrib -y

# Install Nginx (optional - only if you want reverse proxy/SSL)
# sudo apt install nginx -y  # Uncomment if you want to use Nginx
```

## Step 2: Upload Your Code

### Option A: Using Git (Recommended)

```bash
# On your server
cd /var/www
sudo git clone https://github.com/talal-ai/rubai-ragbot-backend.git
sudo chown -R www-data:www-data rubai-ragbot-backend
```

### Option B: Using SCP/SFTP

```bash
# From your local machine
scp -r . user@your-server-ip:/tmp/rubai-ragbot-backend

# On your server
sudo mv /tmp/rubai-ragbot-backend /var/www/
sudo chown -R www-data:www-data /var/www/rubai-ragbot-backend
```

## Step 3: Set Up Environment

```bash
cd /var/www/rubai-ragbot-backend

# Create virtual environment
sudo -u www-data python3 -m venv venv
sudo -u www-data venv/bin/pip install --upgrade pip
sudo -u www-data venv/bin/pip install -r requirements.txt

# Create .env file
sudo -u www-data cp .env.example .env
sudo -u www-data nano .env  # Edit with your configuration
```

## Step 4: Install Systemd Service

```bash
# Copy service file
sudo cp deploy/rubai-backend.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable service (starts on boot)
sudo systemctl enable rubai-backend

# Start service
sudo systemctl start rubai-backend

# Check status
sudo systemctl status rubai-backend
```

## Step 5: Configure Firewall (Required if not using Nginx)

If you're running the app directly without Nginx, you need to allow port 5555:

```bash
# Allow port 5555
sudo ufw allow 5555/tcp

# Or allow from specific IP only (more secure)
sudo ufw allow from your-ip-address to any port 5555
```

**Note:** Nginx is completely optional! Your FastAPI app runs directly on port 8000 and can be accessed without any reverse proxy.

## Step 6: Configure Nginx (Optional - Only if you want a reverse proxy)

Nginx is NOT required. You can access your app directly at `http://your-server-ip:8000`

If you want to use Nginx (for SSL, custom domain, load balancing, etc.):

```bash
# Copy nginx configuration
sudo cp deploy/nginx.conf /etc/nginx/sites-available/rubai-backend

# Edit the server_name
sudo nano /etc/nginx/sites-available/rubai-backend

# Enable site
sudo ln -s /etc/nginx/sites-available/rubai-backend /etc/nginx/sites-enabled/

# Test nginx configuration
sudo nginx -t

# Restart nginx
sudo systemctl restart nginx
```

## Step 7: Set Up SSL with Let's Encrypt (Optional - Only if using Nginx)

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx -y

# Get SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal is set up automatically
```

## Useful Commands

### Service Management

```bash
# Start service
sudo systemctl start rubai-backend

# Stop service
sudo systemctl stop rubai-backend

# Restart service
sudo systemctl restart rubai-backend

# Check status
sudo systemctl status rubai-backend

# View logs
sudo journalctl -u rubai-backend -f
```

### Update Application

```bash
cd /var/www/rubai-ragbot-backend

# Pull latest changes (if using git)
sudo -u www-data git pull

# Update dependencies
sudo -u www-data venv/bin/pip install -r requirements.txt

# Restart service
sudo systemctl restart rubai-backend
```

### Check Application

```bash
# Test if app is running
curl http://localhost:5555/docs

# Check if port is listening
sudo netstat -tlnp | grep 5555
```

## Troubleshooting

### Service won't start

```bash
# Check logs
sudo journalctl -u rubai-backend -n 50

# Check permissions
ls -la /var/www/rubai-ragbot-backend

# Verify .env file exists
ls -la /var/www/rubai-ragbot-backend/.env
```

### Port already in use

```bash
# Find what's using port 8000
sudo lsof -i :8000

# Kill the process or change port in service file
```

### Permission errors

```bash
# Fix ownership
sudo chown -R www-data:www-data /var/www/rubai-ragbot-backend

# Fix permissions
sudo chmod -R 755 /var/www/rubai-ragbot-backend
```

## Security Checklist

- [ ] Change default passwords
- [ ] Set up firewall (UFW)
- [ ] Configure SSL/HTTPS
- [ ] Restrict .env file permissions (chmod 600)
- [ ] Keep system updated
- [ ] Set up log rotation
- [ ] Configure fail2ban (optional)

## Firewall Setup

### Without Nginx (Direct Access)

```bash
# Allow SSH
sudo ufw allow 22/tcp

# Allow direct access to FastAPI app
sudo ufw allow 5555/tcp

# Enable firewall
sudo ufw enable
```

### With Nginx (Reverse Proxy)

```bash
# Allow SSH
sudo ufw allow 22/tcp

# Allow HTTP/HTTPS (for Nginx)
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Enable firewall
sudo ufw enable
```

## File Structure

```
/var/www/rubai-ragbot-backend/
├── main.py
├── requirements.txt
├── .env
├── venv/
├── deploy/
│   ├── rubai-backend.service
│   ├── nginx.conf
│   └── deploy.sh
└── ... (other app files)
```

