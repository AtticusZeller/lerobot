#!/usr/bin/env bash
# =============================================================================
# post-create.sh — runs once after the dev container is first created
# Clones private dotfiles, wires GH_TOKEN, then delegates to install.sh
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

# ---- GH_TOKEN: extract from hosts.yml and inject into shell (keyring unavailable in containers) ----
if [ -f "$DOTFILES_DIR/.config/gh/hosts.yml" ]; then
    GH_TOKEN_FROM_FILE=$(grep 'oauth_token:' "$DOTFILES_DIR/.config/gh/hosts.yml" | head -1 | awk '{print $2}')
    if [ -n "$GH_TOKEN_FROM_FILE" ]; then
        echo "export GH_TOKEN=\"$GH_TOKEN_FROM_FILE\"" >> "$HOME/.zshrc"
        export GH_TOKEN="$GH_TOKEN_FROM_FILE"
        echo "  gh: GH_TOKEN injected into .zshrc"
    fi
fi

# ---- Deploy all configs via install.sh (canonical deployer) ----
echo "Running install.sh..."
bash "$DOTFILES_DIR/install.sh"

# ---- Verify auth status ----
echo ""
echo "=== Auth Status ==="
gh auth status 2>&1 && echo "  GitHub CLI: OK" || echo "  GitHub CLI: NOT authenticated"
hf whoami 2>&1 && echo "  HuggingFace: OK" || echo "  HuggingFace: NOT authenticated"
wandb status 2>&1 | head -2 || true
claude --version 2>&1 && echo "  Claude Code: installed" || echo "  Claude Code: NOT found"

echo ""
echo "Done! Restart shell: exec zsh"
