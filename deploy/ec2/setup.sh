#!/bin/bash
set -euo pipefail

# ============================================================
# ZyncJobs AI Service — EC2 Setup Script
# Automates: https://zyncjobs.com/docs/ec2-setup (25 steps)
# ============================================================
# Usage: sudo bash setup.sh [ollama_model]
#   ollama_model: llama3, mistral, qwen2.5 (default: qwen2.5)
# ============================================================

MODEL="${1:-qwen2.5}"
DOMAIN="${DOMAIN:-api.zyncjobs.com}"
EC2_IP="${EC2_IP:-$(curl -s http://checkip.amazonaws.com)}"

echo "=========================================="
echo " ZyncJobs AI Service — EC2 Bootstrap"
echo " Domain: $DOMAIN"
echo " Model:  $MODEL"
echo "=========================================="

# Step 4: Update OS
echo "[1/25] Updating OS packages..."
apt update && apt upgrade -y

# Step 5: Install common packages
echo "[2/25] Installing system packages..."
apt install -y curl wget git vim zip unzip htop build-essential software-properties-common nginx certbot python3-certbot-nginx

# Step 6: Install Python
echo "[3/25] Installing Python..."
apt install -y python3 python3-pip python3-venv

# Step 7: Create project folder
echo "[4/25] Creating /opt/ai-platform..."
mkdir -p /opt/ai-platform
chown ubuntu:ubuntu /opt/ai-platform

# Step 8-9: Create virtual env
echo "[5/25] Creating Python virtual environment..."
cd /opt/ai-platform
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Step 10: Install AI packages
echo "[6/25] Installing Python packages..."
pip install fastapi uvicorn langgraph langchain langchain-community langchain-ollama pydantic redis qdrant-client psycopg2-binary python-dotenv httpx orjson celery aioboto3 weasyprint jinja2

# Step 11-14: Install and configure Ollama
echo "[7/25] Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh
systemctl enable ollama
systemctl start ollama
sleep 3

echo "[8/25] Configuring Ollama..."
mkdir -p /etc/systemd/system/ollama.service.d
cat > /etc/systemd/system/ollama.service.d/override.conf << 'EOF'
[Service]
Environment="OLLAMA_HOST=127.0.0.1:11434"
Environment="OLLAMA_KEEP_ALIVE=24h"
EOF
systemctl daemon-reload
systemctl restart ollama

# Step 14-15: Download model
echo "[9/25] Downloading model: $MODEL..."
ollama pull "$MODEL"
ollama pull nomic-embed-text

# Step 16: Test Ollama
echo "[10/25] Testing Ollama..."
sleep 2
curl -s http://localhost:11434/api/tags | head -c 200
echo ""

# Step 17-18: Clone and setup application
echo "[11/25] Cloning AI service repository..."
cd /opt/ai-platform
git clone https://github.com/zyncjobs/ai-service.git app 2>/dev/null || {
  echo "Git clone failed — copying from existing source..."
  mkdir -p app
}

# Step 19: Copy .env
echo "[12/25] Setting up environment..."
if [ ! -f /opt/ai-platform/.env ]; then
  cat > /opt/ai-platform/.env << EOF
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_DEFAULT_MODEL=$MODEL
BACKEND_API_URL=http://localhost:5000
ENVIRONMENT=production
RATE_LIMIT_ENABLED=true
CORS_ORIGINS=https://www.zyncjobs.com,https://zyncjobs.com,https://api.zyncjobs.com
LOG_LEVEL=info
EOF
fi

# Step 20: Create systemd service
echo "[13/25] Creating AI service systemd unit..."
cat > /etc/systemd/system/zyncjobs-ai.service << 'EOF'
[Unit]
Description=ZyncJobs AI Service
After=network.target ollama.service
Wants=ollama.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/ai-platform/app
Environment="PATH=/opt/ai-platform/venv/bin"
ExecStart=/opt/ai-platform/venv/bin/uvicorn recruitment_ai.api.main:app --host 0.0.0.0 --port 8001 --workers 4 --timeout-keep-alive 300
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Step 21: Start service
echo "[14/25] Starting AI service..."
systemctl daemon-reload
systemctl enable zyncjobs-ai
systemctl start zyncjobs-ai

# Step 22: Configure Nginx
echo "[15/25] Configuring Nginx reverse proxy..."
cat > /etc/nginx/sites-available/ai-platform << 'EOF'
server {
    listen 80;
    server_name _;

    client_max_body_size 50M;
    keepalive_timeout 300;

    location / {
        proxy_pass http://127.0.0.1:8001;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }

    location /health {
        proxy_pass http://127.0.0.1:8001/health;
        proxy_http_version 1.1;
    }

    location /metrics {
        proxy_pass http://127.0.0.1:8001/metrics;
        allow 10.0.0.0/8;
        allow 172.16.0.0/12;
        allow 192.168.0.0/16;
        deny all;
    }

    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
}
EOF

ln -sf /etc/nginx/sites-available/ai-platform /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl restart nginx

# Step 23: Install SSL (if domain resolves to this server)
echo "[16/25] Attempting SSL certificate..."
if dig +short "$DOMAIN" 2>/dev/null | grep -q "$EC2_IP" 2>/dev/null; then
    certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos --email admin@zyncjobs.com || true
else
    echo "  Domain $DOMAIN does not point to $EC2_IP — skipping SSL"
    echo "  Run later: certbot --nginx -d $DOMAIN"
fi

# Step 24: Verify services
echo "[17/25] Verifying services..."
for svc in ollama zyncjobs-ai nginx; do
    if systemctl is-active --quiet "$svc"; then
        echo "  ✅ $svc — running"
    else
        echo "  ❌ $svc — NOT running"
    fi
done

# Step 25: Health check
echo "[18/25] Running health checks..."
sleep 3
curl -s http://localhost:8001/health
echo ""
curl -s http://localhost/health
echo ""

echo ""
echo "=========================================="
echo " ✅ Setup complete!"
echo " API:       http://$EC2_IP"
echo " Health:    http://$EC2_IP/health"
echo " Model:     $MODEL"
echo " Logs:      journalctl -u zyncjobs-ai -f"
echo "=========================================="
