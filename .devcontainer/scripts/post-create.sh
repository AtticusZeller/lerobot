#!/usr/bin/env bash
# =============================================================================
# post-create.sh — runs once after the dev container is first created
# Clones private dotfiles and wires up all configs + auth tokens
# =============================================================================
set -euo pipefail

DOTFILES_DIR="$HOME/.dotfiles"
DOTFILES_REPO="AtticusZeller/dotfiles-private"

# ---- Clone dotfiles (first time only) ----
# PAT_TOKEN passed via: docker run -e PAT_TOKEN=ghp_xxx ...
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

# ---- Symlink config files ----
echo "Linking dotfiles..."

# Zsh (server-specific template)
if [ -f "$DOTFILES_DIR/.zshrc.server" ]; then
    ln -sf "$DOTFILES_DIR/.zshrc.server" "$HOME/.zshrc"
else
    ln -sf "$DOTFILES_DIR/.zshrc" "$HOME/.zshrc"
fi

ln -sf "$DOTFILES_DIR/.p10k.zsh" "$HOME/.p10k.zsh"

# Tmux personal overrides
if [ -f "$DOTFILES_DIR/.config/tmux/tmux.conf.local" ]; then
    mkdir -p "$HOME/.config/tmux"
    ln -sf "$DOTFILES_DIR/.config/tmux/tmux.conf.local" "$HOME/.config/tmux/tmux.conf.local"
fi

# Git global config
if [ -f "$DOTFILES_DIR/.gitconfig" ]; then
    ln -sf "$DOTFILES_DIR/.gitconfig" "$HOME/.gitconfig"
fi

# Git global ignore
if [ -f "$DOTFILES_DIR/.config/git/ignore" ]; then
    mkdir -p "$HOME/.config/git"
    ln -sf "$DOTFILES_DIR/.config/git/ignore" "$HOME/.config/git/ignore"
fi

# SSH config (copy, not symlink — SSH is picky about permissions)
if [ -d "$DOTFILES_DIR/.ssh" ]; then
    mkdir -p "$HOME/.ssh" && chmod 700 "$HOME/.ssh"
    cp "$DOTFILES_DIR/.ssh/config" "$HOME/.ssh/config" 2>/dev/null || true
    cp "$DOTFILES_DIR/.ssh/id_ed25519" "$HOME/.ssh/id_ed25519" 2>/dev/null || true
    cp "$DOTFILES_DIR/.ssh/id_ed25519.pub" "$HOME/.ssh/id_ed25519.pub" 2>/dev/null || true
    chmod 600 "$HOME/.ssh/id_ed25519" 2>/dev/null || true
    chmod 644 "$HOME/.ssh/id_ed25519.pub" 2>/dev/null || true
    chmod 644 "$HOME/.ssh/config" 2>/dev/null || true
fi

# ---- Migrate auth / credentials ----
echo "Migrating auth configs..."

# HuggingFace
if [ -f "$DOTFILES_DIR/.cache/huggingface/token" ]; then
    mkdir -p "$HOME/.cache/huggingface"
    cp "$DOTFILES_DIR/.cache/huggingface/token" "$HOME/.cache/huggingface/token"
    # Also copy stored_tokens if present
    cp "$DOTFILES_DIR/.cache/huggingface/stored_tokens" "$HOME/.cache/huggingface/stored_tokens" 2>/dev/null || true
fi

# GitHub CLI — use GH_TOKEN env var (keyring not available in containers)
# Token is stored in dotfiles .config/gh/hosts.yml (exported by sync.sh)
if [ -f "$DOTFILES_DIR/.config/gh/hosts.yml" ]; then
    GH_TOKEN_FROM_FILE=$(grep 'oauth_token:' "$DOTFILES_DIR/.config/gh/hosts.yml" | head -1 | awk '{print $2}')
    if [ -n "$GH_TOKEN_FROM_FILE" ]; then
        echo "export GH_TOKEN=\"$GH_TOKEN_FROM_FILE\"" >> "$HOME/.zshrc"
        export GH_TOKEN="$GH_TOKEN_FROM_FILE"
    fi
fi

# Claude Code
if [ -d "$DOTFILES_DIR/.claude" ]; then
    cp -r "$DOTFILES_DIR/.claude/" "$HOME/.claude/"
fi

# wandb (.netrc contains API key)
if [ -f "$DOTFILES_DIR/.netrc" ]; then
    cp "$DOTFILES_DIR/.netrc" "$HOME/.netrc"
    chmod 600 "$HOME/.netrc"
fi

# wandb settings
if [ -f "$DOTFILES_DIR/.config/wandb/settings" ]; then
    mkdir -p "$HOME/.config/wandb"
    cp "$DOTFILES_DIR/.config/wandb/settings" "$HOME/.config/wandb/settings"
fi

# ---- Verify auth status ----
echo ""
echo "=== Auth Status ==="
gh auth status 2>&1 && echo "  GitHub CLI: OK" || echo "  GitHub CLI: NOT authenticated (run: gh auth login)"
huggingface-cli whoami 2>&1 && echo "  HuggingFace: OK" || echo "  HuggingFace: NOT authenticated"
wandb status 2>&1 | head -2 || true
claude --version 2>&1 && echo "  Claude Code: installed" || echo "  Claude Code: NOT found"

echo ""
echo "Done! Restart shell: exec zsh"
