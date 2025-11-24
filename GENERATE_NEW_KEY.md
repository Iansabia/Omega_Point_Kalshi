# Generate New Kalshi API Key

## ✅ Completed Steps

1. ✅ Old API key revoked on Kalshi
2. ✅ Private key removed from git history
3. ✅ Private key file deleted from working directory
4. ✅ `.gitignore` updated to block credentials
5. ✅ Changes force pushed to GitHub
6. ✅ Pre-commit hook installed

---

## 🔑 Next Steps: Generate New API Key

### Step 1: Create Secure Directory for Keys

```bash
# Create directory outside project
mkdir -p ~/.ssh/kalshi
chmod 700 ~/.ssh/kalshi

# Verify it was created
ls -la ~/.ssh/kalshi
```

### Step 2: Generate New API Key on Kalshi

1. Go to https://kalshi.com and log in
2. Navigate to: **Account Settings** → **API** → **API Keys**
3. Click **"Create New API Key"** or **"Generate API Key"**
4. **IMPORTANT**: Download the private key file immediately
   - Kalshi will only show it once!
   - Save it as: `kalshi_private_key_new.pem`

### Step 3: Store Private Key Securely

```bash
# Move downloaded key to secure location
mv ~/Downloads/kalshi_private_key_new.pem ~/.ssh/kalshi/kalshi_private_key.pem

# Set proper permissions (owner read/write only)
chmod 600 ~/.ssh/kalshi/kalshi_private_key.pem

# Verify permissions
ls -la ~/.ssh/kalshi/kalshi_private_key.pem
# Should show: -rw------- (600)
```

### Step 4: Copy API Key ID

When you created the key on Kalshi, you should have received an **API Key ID**. It looks like:
```
KALSHI_API_KEY_abc123def456...
```

Copy this ID - you'll need it for the next step.

### Step 5: Update .env File

```bash
# Edit .env file
nano .env

# Update these lines:
KALSHI_API_KEY_ID="YOUR_NEW_KEY_ID_HERE"
KALSHI_PRIVATE_KEY_PATH="/Users/jaredmarcus/.ssh/kalshi/kalshi_private_key.pem"

# Save and exit (Ctrl+X, then Y, then Enter)
```

**Example .env:**
```bash
# Kalshi API Configuration
KALSHI_API_KEY_ID="KALSHI_API_KEY_abc123def456..."
KALSHI_PRIVATE_KEY_PATH="/Users/jaredmarcus/.ssh/kalshi/kalshi_private_key.pem"

# Alternative: Email/Password (not recommended, only as fallback)
# KALSHI_EMAIL=your_email@example.com
# KALSHI_PASSWORD=your_password
```

### Step 6: Test New API Key

```bash
# Test the new key with a simple balance check
PYTHONPATH=. ./venv/bin/python3 -c "
from src.execution.kalshi_client import KalshiClient
print('Testing new API key...')
client = KalshiClient()
try:
    balance = client.get_balance()
    print('✅ Success! API key works.')
    print(f'Balance: {balance}')
except Exception as e:
    print(f'❌ Error: {e}')
"
```

**Expected output:**
```
Using API key authentication (key ID: KALSHI_API...)
Testing new API key...
✅ Success! API key works.
Balance: {'balance': 10000, ...}
```

### Step 7: Test Full Integration

```bash
# Test circuit breaker and audit logging
PYTHONPATH=. ./venv/bin/python3 scripts/test_circuit_breaker_audit_integration.py
```

---

## 🔒 Security Checklist

Before you're done, verify:

- [ ] ✅ Old API key revoked on Kalshi
- [ ] ✅ New private key stored in `~/.ssh/kalshi/` (NOT in project directory)
- [ ] ✅ Private key permissions set to 600 (owner read/write only)
- [ ] ✅ `.env` file updated with new key ID and path
- [ ] ✅ New key tested and working
- [ ] ✅ Pre-commit hook preventing future leaks
- [ ] ✅ `.gitignore` blocking `*.pem` files

**Test the pre-commit hook:**
```bash
# Try to commit a .pem file (should be blocked)
touch test_key.pem
git add test_key.pem
git commit -m "test"
# Expected: ❌ ERROR: Attempting to commit private key file!

# Clean up
rm test_key.pem
```

---

## 📍 Key Locations

| Item | Location |
|------|----------|
| **Private Key (SECURE)** | `~/.ssh/kalshi/kalshi_private_key.pem` |
| **Environment Variables** | `.env` (project root, git-ignored) |
| **Pre-commit Hook** | `.git/hooks/pre-commit` |
| **Remediation Guide** | `SECURITY_INCIDENT_REMEDIATION.md` |
| **Integration Docs** | `docs/CIRCUIT_BREAKER_AUDIT_LOG_INTEGRATION.md` |

---

## 🚨 What NOT To Do

❌ **DO NOT** store keys in:
- Project directory
- Any subdirectory of the project
- Any directory tracked by git
- Cloud-synced folders (Dropbox, Google Drive, iCloud)
- Shared network drives

❌ **DO NOT** hardcode keys in:
- Python files
- Configuration files (except .env, which is git-ignored)
- Scripts
- Jupyter notebooks

✅ **DO** store keys in:
- `~/.ssh/kalshi/` (recommended)
- System keychain/credential manager
- Environment variables loaded from `.env`

---

## 🆘 Troubleshooting

### Issue: "Private key file not found"

```bash
# Check if file exists
ls -la ~/.ssh/kalshi/kalshi_private_key.pem

# Check .env path matches
cat .env | grep KALSHI_PRIVATE_KEY_PATH

# Fix: Ensure path in .env matches actual file location
```

### Issue: "Permission denied" when reading key

```bash
# Fix permissions
chmod 600 ~/.ssh/kalshi/kalshi_private_key.pem

# Verify
ls -la ~/.ssh/kalshi/kalshi_private_key.pem
```

### Issue: "API authentication failed"

1. Verify key ID is correct:
   ```bash
   cat .env | grep KALSHI_API_KEY_ID
   ```
2. Check key format:
   ```bash
   head -1 ~/.ssh/kalshi/kalshi_private_key.pem
   # Should show: -----BEGIN RSA PRIVATE KEY-----
   ```
3. Verify key is not revoked on Kalshi website

### Issue: "Pre-commit hook not working"

```bash
# Re-install hook
chmod +x .git/hooks/pre-commit

# Test it
touch test.pem
git add test.pem
git commit -m "test"
# Should block the commit
```

---

## 📞 Need Help?

- **Kalshi Support**: support@kalshi.com
- **Documentation**: Read `SECURITY_INCIDENT_REMEDIATION.md`
- **Integration Tests**: Run `scripts/test_circuit_breaker_audit_integration.py`

---

**Last Updated**: November 24, 2025
