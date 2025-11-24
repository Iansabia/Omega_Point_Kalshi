#!/bin/bash
#
# Security Incident Remediation Script
# Removes accidentally committed private key from git history
#
# Usage: ./scripts/fix_leaked_key.sh
#

set -e  # Exit on error

echo "=================================================="
echo "  SECURITY INCIDENT REMEDIATION"
echo "  Removing leaked private key from git history"
echo "=================================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
LEAKED_FILE="kalshi_private_key.pem"
BACKUP_DIR="../Omega_Point_Kalshi_BACKUP_$(date +%Y%m%d_%H%M%S)"

echo -e "${YELLOW}⚠️  WARNING: This script will rewrite git history!${NC}"
echo ""
echo "Before proceeding, you MUST:"
echo "  1. ✅ Revoke the compromised API key on Kalshi"
echo "  2. ✅ Ensure no one else is working on this repo"
echo "  3. ✅ Understand that commit hashes will change"
echo ""
read -p "Have you completed steps 1-2 above? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo -e "${RED}❌ Aborted. Please revoke the API key first.${NC}"
    exit 1
fi

echo ""
echo "Step 1: Creating backup..."
if [ -d "$BACKUP_DIR" ]; then
    echo -e "${YELLOW}⚠️  Backup already exists: $BACKUP_DIR${NC}"
else
    cp -r . "$BACKUP_DIR"
    echo -e "${GREEN}✅ Backup created: $BACKUP_DIR${NC}"
fi

echo ""
echo "Step 2: Checking if private key file exists..."
if [ -f "$LEAKED_FILE" ]; then
    echo -e "${YELLOW}⚠️  Private key file found: $LEAKED_FILE${NC}"
    ls -lh "$LEAKED_FILE"
else
    echo -e "${GREEN}✅ Private key file not in working directory${NC}"
fi

echo ""
echo "Step 3: Checking git history for leaked key..."
if git log --all --full-history --oneline -- "$LEAKED_FILE" | grep -q .; then
    echo -e "${RED}❌ Private key found in git history:${NC}"
    git log --all --full-history --oneline -- "$LEAKED_FILE"
else
    echo -e "${GREEN}✅ Private key not found in git history${NC}"
    echo "Nothing to clean. Exiting."
    exit 0
fi

echo ""
echo "Step 4: Installing git-filter-repo..."
if command -v git-filter-repo &> /dev/null; then
    echo -e "${GREEN}✅ git-filter-repo already installed${NC}"
else
    echo "Installing git-filter-repo..."
    pip3 install git-filter-repo
    echo -e "${GREEN}✅ git-filter-repo installed${NC}"
fi

echo ""
echo -e "${YELLOW}⚠️  About to rewrite git history...${NC}"
echo "This will:"
echo "  - Remove $LEAKED_FILE from ALL commits"
echo "  - Change commit hashes"
echo "  - Require force push to GitHub"
echo ""
read -p "Continue? (yes/no): " confirm2

if [ "$confirm2" != "yes" ]; then
    echo -e "${RED}❌ Aborted${NC}"
    exit 1
fi

echo ""
echo "Step 5: Removing private key from git history..."

# Use git filter-repo to remove the file
git filter-repo --invert-paths --path "$LEAKED_FILE" --force

echo -e "${GREEN}✅ Private key removed from git history${NC}"

echo ""
echo "Step 6: Verifying removal..."
if git log --all --full-history --oneline -- "$LEAKED_FILE" | grep -q .; then
    echo -e "${RED}❌ ERROR: Private key still in history!${NC}"
    exit 1
else
    echo -e "${GREEN}✅ Verified: Private key removed from history${NC}"
fi

echo ""
echo "Step 7: Deleting private key from working directory..."
if [ -f "$LEAKED_FILE" ]; then
    rm -f "$LEAKED_FILE"
    echo -e "${GREEN}✅ Deleted: $LEAKED_FILE${NC}"
else
    echo -e "${GREEN}✅ Private key already removed from working directory${NC}"
fi

echo ""
echo "Step 8: Checking .gitignore..."
if grep -q "*.pem" .gitignore; then
    echo -e "${GREEN}✅ .gitignore already blocks .pem files${NC}"
else
    echo -e "${YELLOW}⚠️  Updating .gitignore to block private keys${NC}"
    cat >> .gitignore << 'EOF'

# Private keys and credentials (NEVER COMMIT!)
*.pem
*.key
*private_key*
kalshi_private_key.pem
EOF
    echo -e "${GREEN}✅ .gitignore updated${NC}"
fi

echo ""
echo "=================================================="
echo -e "${GREEN}✅ Git history cleaned successfully!${NC}"
echo "=================================================="
echo ""
echo "NEXT STEPS:"
echo ""
echo "1. Stage and commit .gitignore changes:"
echo "   git add .gitignore"
echo "   git commit -m 'Security: Update .gitignore to block credentials'"
echo ""
echo "2. Force push to GitHub (THIS WILL REWRITE HISTORY):"
echo "   git push --force-with-lease origin main"
echo ""
echo "3. Generate new API key on Kalshi:"
echo "   - Go to https://kalshi.com → Settings → API Keys"
echo "   - Create new key"
echo "   - Store in ~/.ssh/kalshi/ (NOT in project directory)"
echo ""
echo "4. Update .env with new key path"
echo ""
echo "5. Test new key:"
echo "   PYTHONPATH=. ./venv/bin/python3 -c 'from src.execution.kalshi_client import KalshiClient; print(KalshiClient().get_balance())'"
echo ""
echo -e "${YELLOW}⚠️  IMPORTANT: Anyone who has cloned this repo will need to re-clone it${NC}"
echo ""
echo "Backup saved to: $BACKUP_DIR"
echo ""
