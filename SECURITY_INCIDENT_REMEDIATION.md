# 🚨 SECURITY INCIDENT REMEDIATION - RSA Private Key Exposed

**Incident Date**: November 24, 2025
**Severity**: CRITICAL
**Status**: IN PROGRESS

---

## Incident Summary

**What Happened:**
- RSA private key (`kalshi_private_key.pem`) was accidentally committed to git
- Key was pushed to GitHub in commit `4ef6f96bafa93883616f86e3cd2b50a516acb54f`
- Key is associated with Kalshi API credentials
- **Repository is PUBLIC** (https://github.com/Iansabia/Omega_Point_Kalshi)

**Impact:**
- **CRITICAL**: Anyone with access to the GitHub repository can:
  - Download your private key
  - Use it to authenticate to Kalshi API as you
  - Place trades on your behalf
  - Access your account balance
  - View your trading history

---

## IMMEDIATE ACTIONS (DO NOW!)

### ✅ Step 1: Revoke Compromised API Key (HIGHEST PRIORITY)

**You must do this manually RIGHT NOW:**

1. Go to https://kalshi.com and log in
2. Navigate to: **Account Settings** → **API Keys** → **Manage Keys**
3. Find the API key associated with the exposed private key
   - Check your `.env` file for `KALSHI_API_KEY_ID` to identify it
4. Click **"Revoke"** or **"Delete"** next to that key
5. Confirm the revocation

**Why this is critical:**
- Until you revoke this key, anyone can use it to trade on your account
- Revoking the key immediately invalidates the exposed credentials
- This is more important than cleaning up the git history

**Check `.env` file for the key ID:**
```bash
cat .env | grep KALSHI_API_KEY_ID
```

---

### ✅ Step 2: Remove Private Key from Git History

**Run this automated script:**

```bash
# Save this as: scripts/fix_leaked_key.sh
chmod +x scripts/fix_leaked_key.sh
./scripts/fix_leaked_key.sh
```

**What the script does:**
1. Creates a backup of your repository
2. Uses `git filter-repo` to remove `kalshi_private_key.pem` from all commits
3. Updates `.gitignore` to prevent future accidents
4. Prepares for force push to GitHub

**Manual alternative (if script fails):**

```bash
# Install git-filter-repo
pip3 install git-filter-repo

# Remove the file from git history
git filter-repo --invert-paths --path kalshi_private_key.pem --force

# Verify it's gone
git log --all --full-history -- kalshi_private_key.pem
# (Should return nothing)
```

---

### ✅ Step 3: Delete Private Key from Working Directory

```bash
# Delete the compromised key file
rm -f kalshi_private_key.pem

# Verify it's gone
ls -la kalshi_private_key.pem
# (Should return: No such file or directory)
```

---

### ✅ Step 4: Force Push to GitHub (Rewrites History)

**⚠️ WARNING**: This will rewrite history on GitHub. Anyone who has cloned the repo will need to re-clone it.

```bash
# Add your changes
git add .gitignore
git commit -m "Security: Update .gitignore to prevent credential leaks"

# Force push to overwrite GitHub history
git push --force-with-lease origin main

# If that fails, use:
git push --force origin main
```

**After force push:**
- The private key will be removed from GitHub history
- Old commit `4ef6f96` will be rewritten with a new hash
- Anyone with the old clone will have the exposed key in their local history

---

### ✅ Step 5: Generate New API Key Pair

**On Kalshi:**

1. Go to https://kalshi.com → **Account Settings** → **API Keys**
2. Click **"Create New API Key"**
3. Download the new private key file
4. **Save it securely**: `~/keys/kalshi_private_key.pem` (NOT in your project directory!)
5. Set proper permissions: `chmod 600 ~/keys/kalshi_private_key.pem`
6. Copy the API Key ID (you'll need this for `.env`)

**Update your `.env` file:**

```bash
# Edit .env
KALSHI_API_KEY_ID="your_new_key_id_here"
KALSHI_PRIVATE_KEY_PATH="/Users/jaredmarcus/keys/kalshi_private_key.pem"

# DO NOT use paths inside your project directory!
```

**Test the new key:**

```bash
PYTHONPATH=. ./venv/bin/python3 -c "
from src.execution.kalshi_client import KalshiClient
client = KalshiClient()
balance = client.get_balance()
print(f'Balance: {balance}')
"
```

---

## Verification Checklist

After completing all steps, verify:

- [ ] ✅ Old API key has been **revoked** on Kalshi
- [ ] ✅ Private key file **deleted** from working directory
  ```bash
  ls kalshi_private_key.pem
  # Should return: No such file or directory
  ```
- [ ] ✅ Private key **removed from git history**
  ```bash
  git log --all --full-history --oneline -- kalshi_private_key.pem
  # Should return nothing
  ```
- [ ] ✅ `.gitignore` updated to block `*.pem` files
  ```bash
  grep -q "*.pem" .gitignore && echo "PROTECTED" || echo "NOT PROTECTED"
  ```
- [ ] ✅ Changes **force pushed** to GitHub
  ```bash
  git log --oneline -1
  # Should show commit after 4ef6f96
  ```
- [ ] ✅ New API key **generated** and stored securely (outside project directory)
- [ ] ✅ `.env` updated with new key path and ID
- [ ] ✅ New key **tested** successfully

---

## Additional Security Measures

### 1. **GitHub Repository Settings**

If your repository is **public**, consider:
- Making it **private** (Settings → Danger Zone → Change visibility)
- Only share with trusted collaborators
- Enable **secret scanning** (Settings → Security → Secret scanning)

### 2. **Secure Key Storage**

**Best practices:**
```bash
# Create secure directory for keys
mkdir -p ~/.ssh/kalshi
chmod 700 ~/.ssh/kalshi

# Move private key there
mv kalshi_private_key.pem ~/.ssh/kalshi/
chmod 600 ~/.ssh/kalshi/kalshi_private_key.pem

# Update .env to point to secure location
KALSHI_PRIVATE_KEY_PATH="/Users/jaredmarcus/.ssh/kalshi/kalshi_private_key.pem"
```

**Never store keys in:**
- ❌ Project directory
- ❌ Subdirectories of project
- ❌ Any directory tracked by git
- ❌ Cloud-synced directories (Dropbox, Google Drive, etc.)

### 3. **Git Hooks to Prevent Future Leaks**

Create a pre-commit hook:

```bash
# Create: .git/hooks/pre-commit
#!/bin/bash

# Check for private keys before committing
if git diff --cached --name-only | grep -E '\.(pem|key)$'; then
    echo "❌ ERROR: Attempting to commit private key file!"
    echo "Files blocked:"
    git diff --cached --name-only | grep -E '\.(pem|key)$'
    exit 1
fi

# Check for hardcoded credentials
if git diff --cached | grep -iE '(api[_-]?key|secret|password|token|private[_-]?key).*[=:].*[a-zA-Z0-9]{20,}'; then
    echo "⚠️  WARNING: Possible hardcoded credentials detected"
    echo "Review your changes carefully"
    exit 1
fi

exit 0
```

```bash
chmod +x .git/hooks/pre-commit
```

### 4. **Environment Variable Management**

**Never commit `.env` files**. Always use `.env.example`:

```bash
# Create .env.example (safe to commit)
cat > .env.example << 'EOF'
# Kalshi API Configuration
KALSHI_API_KEY_ID=your_api_key_id_here
KALSHI_PRIVATE_KEY_PATH=/path/to/your/private/key.pem

# Alternative: Email/Password (not recommended)
# KALSHI_EMAIL=your_email@example.com
# KALSHI_PASSWORD=your_password
EOF

git add .env.example
git commit -m "Add environment variable template"
```

---

## Post-Incident Monitoring

### Check for Unauthorized Activity

1. **Review Kalshi Account Activity**
   - Log in to https://kalshi.com
   - Go to **Account** → **Activity** or **Trade History**
   - Look for suspicious trades between now and when you revoked the key
   - Check for unusual API access patterns

2. **Monitor Account Balance**
   ```bash
   PYTHONPATH=. ./venv/bin/python3 -c "
   from src.execution.kalshi_client import KalshiClient
   client = KalshiClient()
   balance = client.get_balance()
   print(balance)
   "
   ```

3. **Check API Usage Logs** (if Kalshi provides them)
   - Look for API calls from unknown IP addresses
   - Check for access times that don't match your activity

---

## Timeline of Incident

| Time | Event |
|------|-------|
| Unknown | Private key committed to git in commit `4ef6f96` |
| Unknown | Commit pushed to GitHub (public repository) |
| Nov 24, 2025 | Incident discovered during code review |
| Nov 24, 2025 | Remediation begun |
| **PENDING** | API key revoked on Kalshi |
| **PENDING** | Git history cleaned |
| **PENDING** | New key generated |

---

## Lessons Learned

### What Went Wrong

1. ❌ Private key stored in project directory
2. ❌ `.gitignore` didn't block `*.pem` files
3. ❌ No pre-commit hooks to catch credentials
4. ❌ No secrets scanning enabled on GitHub
5. ❌ No code review before push

### Prevention for Future

1. ✅ Store keys outside project directory
2. ✅ Comprehensive `.gitignore` for credentials
3. ✅ Pre-commit hooks to block sensitive files
4. ✅ Enable GitHub secret scanning
5. ✅ Code review process for all commits
6. ✅ Regular security audits

---

## Contact Information

**If you need help:**
- GitHub Security: security@github.com
- Kalshi Support: support@kalshi.com

**If you suspect unauthorized trading:**
- Contact Kalshi support immediately
- Report the incident
- Request account freeze if necessary

---

## Status Updates

- [ ] **2025-11-24 19:35**: Incident discovered ✅
- [ ] **PENDING**: API key revoked
- [ ] **PENDING**: Git history cleaned
- [ ] **PENDING**: New key generated
- [ ] **PENDING**: Changes force pushed
- [ ] **PENDING**: Verification complete
- [ ] **PENDING**: Incident closed

---

**Last Updated**: November 24, 2025
**Document Version**: 1.0
