# EC2 Deployment Guide

## Quick Summary for EC2

Your QUICK_START.md has been updated for **Amazon Linux 2 on EC2**. Here are the key differences from local development:

---

## 🚀 30-Second EC2 Launch

```bash
# 1. SSH in
ssh -i your-key.pem ec2-user@your-ec2-ip

# 2. Navigate to project
cd ~/local_analytic_chatbot
source venv/bin/activate

# 3. Start server
uvicorn backend.fastapi_app:app --host 0.0.0.0 --port 8000 --reload

# 4. Open browser
http://your-ec2-public-ip:8000
```

---

## Key EC2 Differences

| Local | EC2 |
|-------|-----|
| `brew install` | `sudo yum install` |
| `127.0.0.1:8000` | `0.0.0.0:8000` |
| Localhost only | Public IP access |
| Manual startup | systemd service |
| `CMAKE_ARGS="-DLLAMA_METAL=on"` | Standard llama-cpp-python |

---

## Security Group Setup

**Must allow inbound traffic on port 8000:**

1. Go to AWS Console
2. EC2 Dashboard → Your Instance
3. Click Security Group
4. Edit Inbound Rules
5. Add:
   - Type: Custom TCP
   - Port: 8000
   - Source: 0.0.0.0/0 (or specific IP)

---

## Find Your EC2 Public IP

```bash
# From EC2 console, or:
curl http://169.254.169.254/latest/meta-data/public-ipv4
```

Then visit: **http://<your-public-ip>:8000**

---

## Run in Background

**Option 1: nohup (simple)**
```bash
nohup uvicorn backend.fastapi_app:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &
```

**Option 2: tmux (persistent)**
```bash
tmux new-session -d -s chatbot "source venv/bin/activate && uvicorn backend.fastapi_app:app --host 0.0.0.0 --port 8000"
```

**Option 3: systemd (production, survives reboot)**
See QUICK_START.md for full systemd setup

---

## Verify It's Running

```bash
# Check process
ps aux | grep uvicorn

# Check logs
tail -f server.log

# Check port
sudo lsof -i :8000
```

---

## That's It!

Your chatbot is now running on EC2. Use it just like locally, but with public internet access.

For full setup details, see **QUICK_START.md**
