# =============================================================================
# GPU Dev Container — reusable base for LeRobot / VLA research
# Base: CUDA 12.4 devel on Ubuntu 22.04
# =============================================================================
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# Layer 1: System tools
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs curl wget unzip build-essential openssh-server \
    ca-certificates locales software-properties-common \
    zsh tmux bat fzf ripgrep fd-find tree htop jq \
    fontconfig xz-utils \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# ---------------------------------------------------------------------------
# Layer 2: Zsh + Oh My Zsh + Powerlevel10k + plugins
# (Pure install — .zshrc config comes from dotfiles)
# ---------------------------------------------------------------------------
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

# Nerd Fonts (JetBrainsMono)
RUN mkdir -p /root/.local/share/fonts \
    && cd /root/.local/share/fonts \
    && wget -q https://github.com/ryanoasis/nerd-fonts/releases/latest/download/JetBrainsMono.tar.xz \
    && tar -xf JetBrainsMono.tar.xz \
    && rm JetBrainsMono.tar.xz \
    && fc-cache -fv

# Powerlevel10k theme
RUN git clone --depth=1 https://github.com/romkatv/powerlevel10k.git \
    ${ZSH_CUSTOM:-/root/.oh-my-zsh/custom}/themes/powerlevel10k

# Zsh plugins
RUN git clone --depth=1 https://github.com/zsh-users/zsh-autosuggestions.git \
      ${ZSH_CUSTOM:-/root/.oh-my-zsh/custom}/plugins/zsh-autosuggestions \
    && git clone --depth=1 https://github.com/zsh-users/zsh-syntax-highlighting.git \
      ${ZSH_CUSTOM:-/root/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting \
    && git clone --depth=1 https://github.com/Pilaton/OhMyZsh-full-autoupdate.git \
      ${ZSH_CUSTOM:-/root/.oh-my-zsh/custom}/plugins/ohmyzsh-full-autoupdate \
    && git clone --depth=1 https://github.com/fdellwing/zsh-bat.git \
      ${ZSH_CUSTOM:-/root/.oh-my-zsh/custom}/plugins/zsh-bat

# ---------------------------------------------------------------------------
# Layer 3: Tmux + Oh My Tmux
# ---------------------------------------------------------------------------
RUN git clone --single-branch https://github.com/gpakosz/.tmux.git /root/.local/share/oh-my-tmux \
    && mkdir -p /root/.config/tmux \
    && ln -s /root/.local/share/oh-my-tmux/.tmux.conf /root/.config/tmux/tmux.conf \
    && cp /root/.local/share/oh-my-tmux/.tmux.conf.local /root/.config/tmux/tmux.conf.local

# ---------------------------------------------------------------------------
# Layer 4: UV + Python 3.12
# ---------------------------------------------------------------------------
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
RUN uv python install 3.12 \
    && ln -sf $(uv python find 3.12) /usr/local/bin/python3 \
    && ln -sf $(uv python find 3.12) /usr/local/bin/python

# ---------------------------------------------------------------------------
# Layer 5: UV global tools
# ---------------------------------------------------------------------------
RUN uv tool install nvitop \
    && uv tool install wandb \
    && uv tool install huggingface-hub

# ---------------------------------------------------------------------------
# Layer 6: Node.js (nvm) + Claude Code + cc-switch
# ---------------------------------------------------------------------------
ENV NVM_DIR="/root/.nvm"

RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.4/install.sh | bash \
    && . "$NVM_DIR/nvm.sh" \
    && nvm install 24 \
    && ln -sf "$(which node)" /usr/local/bin/node \
    && ln -sf "$(which npm)" /usr/local/bin/npm \
    && ln -sf "$(which npx)" /usr/local/bin/npx

# Claude Code CLI
RUN curl -fsSL https://claude.ai/install.sh | bash

# cc-switch-cli
RUN curl -fsSL https://github.com/SaladDay/cc-switch-cli/releases/latest/download/install.sh | bash

# ---------------------------------------------------------------------------
# Layer 7: GitHub CLI
# ---------------------------------------------------------------------------
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
      | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
      | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt-get update && apt-get install -y gh \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Layer 8: Git global config + default shell
# ---------------------------------------------------------------------------
RUN git config --global user.name "A.J.Zeller" \
    && git config --global user.email "hello@atticux.me" \
    && git config --global init.defaultBranch main \
    && git lfs install \
    && chsh -s $(which zsh)

WORKDIR /workspace
SHELL ["/bin/zsh", "-c"]
CMD ["zsh"]
