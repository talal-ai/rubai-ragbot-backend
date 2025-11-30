# Deploy Without Nginx - Simple Guide

Your FastAPI backend can run **directly without Nginx**. This is simpler and perfect for:
- Development/testing
- Internal APIs
- When you don't need SSL/HTTPS
- When you don't need a custom domain

## Quick Deployment (No Nginx)

### 1. Upload Your Code

```bash
# From your local machine
scp -r . user@your-server-ip:/tmp/rubai-backend
```

### 2. On Your Server

```bash
# Move to /var/www
sudo mv /tmp/rubai-backend /var/www/rubai-ragbot-backend
cd /var/www/rubai-ragbot-backend

# Create virtual environment
sudo -u www-data python3 -m venv venv
sudo -u www-data venv/bin/pip install -r requirements.txt

# Create .env file
sudo -u www-data cp .env .env.backup  # Backup if exists
sudo -u www-data nano .env  # Edit with your settings
```

### 3. Install Systemd Service

```bash
# Copy service file
sudo cp deploy/rubai-backend.service /etc/systemd/system/

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable rubai-backend
sudo systemctl start rubai-backend

# Check status
sudo systemctl status rubai-backend
```

### 4. Configure Firewall

```bash
# Allow port 5555
sudo ufw allow 5555/tcp

# Or allow from specific IP (more secure)
sudo ufw allow from your-ip-address to any port 5555

# Enable firewall
sudo ufw enable
```

### 5. Access Your App

Your app is now running directly on port 5555:

- **API**: `http://your-server-ip:5555`
- **Docs**: `http://your-server-ip:5555/docs`
- **Health**: `http://your-server-ip:5555/health`

## That's It!

No Nginx needed. Your FastAPI app runs directly and is accessible on port 5555.

## When to Use Nginx

You might want Nginx later if you need:
- ✅ SSL/HTTPS (Let's Encrypt)
- ✅ Custom domain (api.yourdomain.com)
- ✅ Load balancing
- ✅ Static file serving
- ✅ Rate limiting
- ✅ Hide backend port

But for now, **direct access works perfectly fine!**

## Change Port (Optional)

If you want to use a different port (like 80 or 443), edit the service file:

```bash
sudo nano /etc/systemd/system/rubai-backend.service
```

Change the port in the ExecStart line:
```
ExecStart=/var/www/rubai-ragbot-backend/venv/bin/uvicorn main:app --host 0.0.0.0 --port 80
```

Then restart:
```bash
sudo systemctl daemon-reload
sudo systemctl restart rubai-backend
```

## Security Note

When running without Nginx:
- ✅ Your app is directly exposed on port 5555
- ✅ Consider using a firewall to restrict access
- ✅ For production, consider adding Nginx for SSL/HTTPS
- ✅ FastAPI has built-in security, but HTTPS is recommended for production

## Troubleshooting

### Can't access from outside

```bash
# Check if service is running
sudo systemctl status rubai-backend

# Check if port is listening
sudo netstat -tlnp | grep 5555

# Check firewall
sudo ufw status

# Check app logs
sudo journalctl -u rubai-backend -f
```

### Permission errors

```bash
# Fix ownership
sudo chown -R www-data:www-data /var/www/rubai-ragbot-backend

# Fix permissions
sudo chmod -R 755 /var/www/rubai-ragbot-backend
```

