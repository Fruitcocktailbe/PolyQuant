# AWS Deployment Guide for PolyQuant 2.0

**Target**: AWS Lightsail in London for optimal latency to Polymarket

**Expected Latency**:
- AWS London → Polymarket WebSocket: 10-30ms
- AWS London → Polygon RPC (EU): 20-40ms
- **Total p95**: <40ms (under 50ms target! ✓)

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [AWS Lightsail Setup](#aws-lightsail-setup)
3. [Redis Installation](#redis-installation)
4. [Application Deployment](#application-deployment)
5. [Security Configuration](#security-configuration)
6. [Monitoring Setup](#monitoring-setup)
7. [Testing & Validation](#testing--validation)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Accounts
- ✅ AWS Account with billing enabled
- ✅ Polymarket account (for API access)
- ✅ Polygon RPC provider (Alchemy/Infura recommended)
- ✅ GitHub account (for code deployment)

### Local Requirements
- SSH client (Windows: PuTTY/OpenSSH, macOS/Linux: built-in)
- `aws` CLI installed (optional, for automation)
- Git installed

---

## AWS Lightsail Setup

### Step 1: Create Lightsail Instance

1. **Log into AWS Console**
   - Navigate to: https://lightsail.aws.amazon.com/

2. **Create Instance**
   - Click "Create instance"
   - **Region**: Select "London" (eu-west-2)
   - **Platform**: Linux/Unix
   - **OS**: Ubuntu 22.04 LTS
   - **SSH key**: Create new or use existing

3. **Choose Instance Plan**
   - **Minimum**: $40/month (2 vCPU, 8GB RAM, 160GB SSD)
   - **Recommended**: $80/month (4 vCPU, 16GB RAM, 320GB SSD)
     - SCIP solver benefits from extra RAM
     - More CPU = faster opportunity detection

4. **Name Your Instance**
   - Name: `polyquant-prod-london`

5. **Create Instance**
   - Wait 2-3 minutes for provisioning

### Step 2: Configure Static IP

1. **Networking Tab**
   - Click "Create static IP"
   - Attach to your instance
   - **Why**: Consistent outbound IP for whitelisting/monitoring

2. **Note Your IP**
   - Save the static IP for later (e.g., `18.130.XXX.XXX`)

### Step 3: Configure Firewall

1. **Networking → Firewall**
   - **Default Rules**:
     - SSH (TCP/22): Allow from your IP only (security!)
     - HTTP (TCP/80): Optional (for monitoring dashboard)
     - HTTPS (TCP/443): Optional (for monitoring dashboard)

   - **Add Custom Rule** (optional for monitoring):
     - Application: Custom
     - Protocol: TCP
     - Port: 8080 (FastAPI monitoring)
     - Source: Your IP only

**Security Note**: NEVER expose port 22 (SSH) to 0.0.0.0/0 in production!

---

## Redis Installation

### Option A: Local Redis (Simpler, slightly higher latency)

**Recommended for MVP**. Lower complexity, ~1-2ms latency.

```bash
# SSH into your Lightsail instance
ssh -i ~/.ssh/LightsailDefaultKey-eu-west-2.pem ubuntu@18.130.XXX.XXX

# Update system
sudo apt update && sudo apt upgrade -y

# Install Redis
sudo apt install redis-server -y

# Configure Redis for production
sudo nano /etc/redis/redis.conf

# Make these changes:
#   supervised systemd (enable systemd management)
#   maxmemory 2gb (limit memory usage)
#   maxmemory-policy allkeys-lru (evict old keys when full)

# Restart Redis
sudo systemctl restart redis
sudo systemctl enable redis

# Verify
redis-cli ping
# Should return: PONG
```

### Option B: AWS ElastiCache (More complex, slightly lower latency)

**For high-scale production** (~0.5ms latency, more expensive).

1. **Create ElastiCache Cluster**
   - AWS Console → ElastiCache
   - Engine: Redis
   - Version: 7.0 or higher
   - Node type: cache.t3.micro ($15/month)
   - **IMPORTANT**: Same VPC as Lightsail!

2. **Connect Lightsail to VPC**
   - Lightsail → Networking → VPC Peering
   - Enable peering with default VPC

3. **Security Group**
   - Allow port 6379 from Lightsail instance IP

4. **Update Config**
   - In `config.py`, set Redis host to ElastiCache endpoint

---

## Application Deployment

### Step 1: Install System Dependencies

```bash
# SSH into instance
ssh -i ~/.ssh/YourKey.pem ubuntu@18.130.XXX.XXX

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt install python3.11 python3.11-venv python3.11-dev -y

# Install system dependencies
sudo apt install build-essential libssl-dev libffi-dev -y
sudo apt install git curl -y

# Install SCIP Optimization Suite
cd /tmp
wget https://scipopt.org/download/release/SCIPOptSuite-9.0.0-Linux-ubuntu.deb
sudo apt install ./SCIPOptSuite-9.0.0-Linux-ubuntu.deb -y
```

### Step 2: Clone Repository

```bash
# Create app directory
sudo mkdir -p /opt/polyquant
sudo chown ubuntu:ubuntu /opt/polyquant
cd /opt/polyquant

# Clone repository
git clone https://github.com/yourusername/PolyQuant.git .

# Or use SSH if you have keys set up
git clone git@github.com:yourusername/PolyQuant.git .
```

### Step 3: Setup Python Environment

```bash
cd /opt/polyquant

# Create virtual environment
python3.11 -m venv venv

# Activate environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Verify installation
python -c "import pyscipopt; print('SCIP OK')"
python -c "import redis; print('Redis OK')"
```

### Step 4: Configure Environment

```bash
# Create environment file
nano .env

# Add your configuration:
```

```ini
# Polymarket API
POLYMARKET_API_KEY=your_api_key_here
POLYMARKET_SECRET=your_secret_here

# Polygon RPC (Alchemy London recommended)
POLYGON_RPC_URL=https://polygon-mainnet.g.alchemy.com/v2/YOUR_KEY

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Risk Management
MAX_POSITION_SIZE=1000
MAX_DAILY_LOSS=5000
KILL_SWITCH_DRAWDOWN=0.15

# Mode
PAPER_MODE=true  # Set to false for live trading!
```

```bash
# Set correct permissions
chmod 600 .env
```

### Step 5: Create Systemd Service

```bash
# Create service file
sudo nano /etc/systemd/system/polyquant.service
```

```ini
[Unit]
Description=PolyQuant 2.0 Trading System
After=network.target redis.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/polyquant
Environment="PATH=/opt/polyquant/venv/bin"
ExecStart=/opt/polyquant/venv/bin/python -m polyquant.main trade --mode paper
Restart=always
RestartSec=10

# Security hardening
NoNewPrivileges=true
PrivateTmp=true

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=polyquant

[Install]
WantedBy=multi-user.target
```

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service (start on boot)
sudo systemctl enable polyquant

# Start service
sudo systemctl start polyquant

# Check status
sudo systemctl status polyquant

# View logs
sudo journalctl -u polyquant -f
```

---

## Security Configuration

### SSH Hardening

```bash
# Edit SSH config
sudo nano /etc/ssh/sshd_config

# Recommended settings:
#   PermitRootLogin no
#   PasswordAuthentication no
#   PubkeyAuthentication yes
#   Port 2222 (change default port)

# Restart SSH
sudo systemctl restart ssh
```

### Firewall (UFW)

```bash
# Enable UFW
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow SSH (change port if you modified it)
sudo ufw allow 2222/tcp

# Allow monitoring dashboard (optional, your IP only)
sudo ufw allow from YOUR_IP_ADDRESS to any port 8080

# Enable firewall
sudo ufw enable

# Check status
sudo ufw status verbose
```

### Automatic Security Updates

```bash
# Install unattended-upgrades
sudo apt install unattended-upgrades -y

# Enable automatic security updates
sudo dpkg-reconfigure --priority=low unattended-upgrades
```

---

## Monitoring Setup

### Option A: Simple Logging (MVP)

```bash
# View real-time logs
sudo journalctl -u polyquant -f

# View last 100 lines
sudo journalctl -u polyquant -n 100

# Search for errors
sudo journalctl -u polyquant | grep ERROR

# View logs from last hour
sudo journalctl -u polyquant --since "1 hour ago"
```

### Option B: CloudWatch Integration (Recommended)

```bash
# Install CloudWatch agent
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
sudo dpkg -i -E ./amazon-cloudwatch-agent.deb

# Configure agent
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-config-wizard

# Logs to monitor:
#   /var/log/syslog
#   /opt/polyquant/logs/polyquant.log (if you add file logging)

# Start agent
sudo systemctl start amazon-cloudwatch-agent
```

### Option C: Grafana Cloud (Free Tier)

1. **Sign up**: https://grafana.com/
2. **Install Grafana Agent**:

```bash
# Download agent
wget https://github.com/grafana/agent/releases/download/v0.38.1/grafana-agent-linux-amd64.zip
unzip grafana-agent-linux-amd64.zip

# Configure agent (follow Grafana Cloud setup wizard)
sudo nano /etc/grafana-agent.yaml

# Start agent
sudo systemctl start grafana-agent
```

### Key Metrics to Monitor

- **Latency**: p50, p95, p99 (should be <40ms)
- **Opportunity Rate**: Opportunities detected per minute
- **Trade Success Rate**: Filled trades / attempted trades
- **WebSocket Health**: Connection age, message rate
- **Memory Usage**: Should be stable (~500MB-2GB)
- **CPU Usage**: Should be <50% average
- **Redis Hit Rate**: >90% for InitFW cache

---

## Testing & Validation

### Pre-Production Checklist

```bash
# 1. Test Redis connection
redis-cli ping

# 2. Test Polygon RPC
curl -X POST https://polygon-mainnet.g.alchemy.com/v2/YOUR_KEY \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}'

# 3. Test Navigator in paper mode
cd /opt/polyquant
source venv/bin/activate
python -m polyquant.main trade --mode paper --max-ticks 10

# 4. Run E2E latency test (1 hour)
python scripts/test_e2e_latency.py --duration 3600

# 5. Check report
cat latency_report.json

# Success criteria:
#   ✓ p95 < 50ms
#   ✓ p99 < 100ms
#   ✓ No crashes
#   ✓ WebSocket stable
```

### 24-Hour Soak Test

```bash
# Run Navigator for 24 hours in paper mode
python -m polyquant.main trade --mode paper --duration 86400

# Monitor logs
sudo journalctl -u polyquant -f

# After 24 hours, check:
#   - Memory usage (should be stable)
#   - Uptime (should be >99.5% = <7 min downtime)
#   - Latency report (p95 should still be <50ms)
#   - No memory leaks
```

---

## Going Live (Real Money)

**⚠️ WARNING**: Only proceed after successful 24-hour soak test!

### Step 1: Update Configuration

```bash
nano /opt/polyquant/.env

# Change mode
PAPER_MODE=false

# Set conservative position limits for first week
MAX_POSITION_SIZE=100  # Start small!
MAX_DAILY_LOSS=500
KILL_SWITCH_DRAWDOWN=0.10  # More aggressive for small capital
```

### Step 2: Fund Account

- Transfer initial capital to Polymarket account
- **Recommended starting capital**: $1,000-$5,000
- Keep withdrawal threshold high initially

### Step 3: Gradual Ramp-Up

**Week 1**: $100 max position
- Monitor for unexpected behaviors
- Verify trades execute as expected
- Check actual latency vs paper mode

**Week 2**: $500 max position (if Week 1 successful)
- Continue monitoring
- Track actual P&L vs expected

**Week 3**: $1,000 max position (if Week 2 successful)
- Full production mode
- Monitor for MEV attacks, front-running

### Step 4: Daily Monitoring Routine

```bash
# Morning check
sudo systemctl status polyquant
sudo journalctl -u polyquant --since "24 hours ago" | grep ERROR

# Check P&L
# (Add your P&L tracking mechanism here)

# Check kill switch status
redis-cli GET polyquant:kill_switch:active

# Review trades
redis-cli KEYS "polyquant:trades:*"
```

---

## Troubleshooting

### Issue 1: Navigator Won't Start

**Symptoms**: `systemctl status polyquant` shows "failed"

**Debug**:
```bash
# Check logs
sudo journalctl -u polyquant -n 50

# Common causes:
#   - Redis not running: sudo systemctl start redis
#   - Python path wrong: check /etc/systemd/system/polyquant.service
#   - Missing dependencies: source venv/bin/activate && pip install -r requirements.txt
```

### Issue 2: High Latency (>50ms p95)

**Possible causes**:
1. **Network latency**: Test with `ping polymarket.com` and check RPC latency
2. **Redis slow**: Check `redis-cli --latency` (should be <2ms)
3. **CPU throttling**: Check `top` - if CPU >90%, upgrade instance
4. **Memory pressure**: Check `free -h` - if swap used, upgrade instance

### Issue 3: WebSocket Keeps Disconnecting

**Debug**:
```bash
# Check WebSocket health
sudo journalctl -u polyquant | grep "WebSocket"

# Common causes:
#   - Polymarket rate limiting: Reduce subscription frequency
#   - Network instability: Check AWS service health dashboard
#   - Firewall blocking: Check UFW rules
```

### Issue 4: Redis Cache Misses

**Symptoms**: InitFW cache hit rate <50%

**Debug**:
```bash
# Check Redis memory
redis-cli INFO memory

# Check cache keys
redis-cli KEYS "initfw:*"

# Common causes:
#   - Redis restarted (lost cache): Normal, will rebuild
#   - Keys expiring too soon: Check TTL settings
#   - Memory eviction: Increase maxmemory in redis.conf
```

---

## Maintenance

### Daily Tasks
- ✅ Check systemd service status
- ✅ Review error logs
- ✅ Monitor P&L and position sizes
- ✅ Check kill switch hasn't triggered

### Weekly Tasks
- ✅ Review latency metrics (ensure still <50ms p95)
- ✅ Check for system updates: `sudo apt update && sudo apt list --upgradable`
- ✅ Review Redis cache hit rates
- ✅ Backup configuration and .env file

### Monthly Tasks
- ✅ Apply security updates: `sudo apt upgrade -y`
- ✅ Review CloudWatch/Grafana dashboards
- ✅ Analyze trade performance vs benchmarks
- ✅ Consider scaling up instance if needed

---

## Scaling Up

When you're ready to scale beyond $5K capital:

### Vertical Scaling (More Power)
- Upgrade to $160/month (8 vCPU, 32GB RAM)
- Benefits: Faster SCIP solving, more concurrent opportunities

### Horizontal Scaling (Multi-Region)
- Deploy second instance in US East (for US market hours)
- Use Redis Cluster for shared cache
- Route trades to closest instance

### Advanced Optimizations
- Migrate execution layer to Rust
- Use private Polygon RPC (QuickNode, Blast API) for <10ms latency
- Implement pre-signed transaction pool

---

## Support & Resources

- **AWS Lightsail Docs**: https://lightsail.aws.amazon.com/ls/docs
- **PolyQuant GitHub**: https://github.com/yourusername/PolyQuant
- **Polymarket API**: https://docs.polymarket.com/
- **SCIP Documentation**: https://scipopt.org/doc/html/

---

## Cost Breakdown (Monthly)

| Service | Plan | Cost |
|---------|------|------|
| **Lightsail** | 4 vCPU, 16GB RAM | $80 |
| **ElastiCache (optional)** | cache.t3.micro | $15 |
| **Polygon RPC** | Alchemy Growth | $49 |
| **CloudWatch (optional)** | Logs + Metrics | ~$10 |
| **Data Transfer** | ~50GB/month | $5 |
| **Total (Basic)** | | **~$150/month** |
| **Total (Full)** | | **~$160/month** |

**ROI**: If system generates >$500/month profit, infrastructure pays for itself 3×!

---

## Final Checklist Before Live Trading

- [ ] 24-hour soak test passed
- [ ] p95 latency < 50ms verified
- [ ] WebSocket uptime > 99.5%
- [ ] Kill switch tested and working
- [ ] Backups configured (.env, manifests)
- [ ] Monitoring dashboard set up
- [ ] Emergency stop procedure documented
- [ ] Capital allocated ($1K-$5K recommended for start)
- [ ] Paper trading successful for 1+ week

**🚀 Ready to deploy? Start with paper mode, then gradually ramp up!**
