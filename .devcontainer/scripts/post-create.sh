#!/usr/bin/env bash
# =============================================================================
# post-create.sh — runs once after the dev container is first created
# Clones private dotfiles, delegates to install.sh, then wires GH_TOKEN
# =============================================================================
set -euo pipefail

DOTFILES_DIR="$HOME/.dotfiles"
DOTFILES_REPO="AtticusZeller/dotfiles-private"

# ---- Clone dotfiles (first time only) ----
# PAT_TOKEN passed via: devcontainer.json containerEnv -> ${localEnv:PAT_TOKEN}
if [ ! -d "$DOTFILES_DIR" ]; then
    if [ -n "${PAT_TOKEN:-}" ]; then
        echo "Cloning dotfiles via PAT..."
        git clone "https://${PAT_TOKEN}@github.com/${DOTFILES_REPO}.git" "$DOTFILES_DIR"
    else
        echo "WARNING: PAT_TOKEN not set, skipping dotfiles clone."
        echo "Run manually: git clone https://<PAT>@github.com/${DOTFILES_REPO}.git $DOTFILES_DIR"
        exit 0
    fi
fi

# ---- Deploy all configs via install.sh (canonical deployer) ----
# Must run BEFORE GH_TOKEN injection — install.sh symlinks .zshrc.server to ~/.zshrc
echo "Running install.sh..."
bash "$DOTFILES_DIR/install.sh"

# ---- GH_TOKEN: extract from hosts.yml and append to .zshrc (keyring unavailable in containers) ----
# Runs AFTER install.sh so the .zshrc symlink is already in place
if [ -f "$DOTFILES_DIR/.config/gh/hosts.yml" ]; then
    GH_TOKEN_FROM_FILE=$(grep 'oauth_token:' "$DOTFILES_DIR/.config/gh/hosts.yml" | head -1 | awk '{print $2}')
    if [ -n "$GH_TOKEN_FROM_FILE" ]; then
        printf '\nexport GH_TOKEN="%s"\n' "$GH_TOKEN_FROM_FILE" >> "$HOME/.zshrc"
        export GH_TOKEN="$GH_TOKEN_FROM_FILE"
        echo "  gh: GH_TOKEN injected into .zshrc"
    fi
fi

# ---- Verify tools installed ----
echo ""
echo "=== Tool Verification ==="
for cmd in git gh hf claude python3 node uv nvcc; do
    if command -v "$cmd" &>/dev/null; then
        echo "  $cmd: OK"
    else
        echo "  $cmd: NOT found"
    fi
done
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
    && echo "  GPU: OK" || echo "  GPU: NOT available"

# ---- Verify auth ----
echo ""
echo "=== Auth Verification ==="
gh api user --jq .login 2>&1 && echo "  GitHub CLI: OK" || echo "  GitHub CLI: NOT authenticated"
hf auth whoami 2>&1 | head -3 || true
wandb login 2>&1 | head -3 || true
claude auth status 2>&1 | head -5 || true

echo ""
echo "Done! Restart shell: exec zsh"
