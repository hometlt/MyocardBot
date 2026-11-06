#!/usr/bin/env bash
# One-time provisioning of an Ubuntu 24.04 server for Miocard Bot.
# Run as root on the server:
#   bash server-setup.sh
set -euo pipefail

DEPLOY_DIR=/srv/miocard
REPO_DIR=/srv/miocard.git
BOT_UID=10001

if [ "$(id -u)" -ne 0 ]; then
    echo "Run this script as root." >&2
    exit 1
fi

echo "==> Installing Docker and the Compose plugin"
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y ca-certificates curl git

if command -v docker >/dev/null 2>&1; then
    # Ubuntu 24.04 images often ship docker.io already; adding the upstream
    # repository on top of it would force a replacement of the running engine.
    echo "Docker is already installed: $(docker --version)"
    if ! docker compose version >/dev/null 2>&1; then
        apt-get install -y docker-compose-v2
    fi
else
    install -m 0755 -d /etc/apt/keyrings
    if [ ! -f /etc/apt/keyrings/docker.asc ]; then
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
        chmod a+r /etc/apt/keyrings/docker.asc
    fi
    UBUNTU_CODENAME=$(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
    cat > /etc/apt/sources.list.d/docker.list <<EOF
deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${UBUNTU_CODENAME} stable
EOF
    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi
systemctl enable --now docker
docker compose version

echo "==> Ensuring at least 2 GiB of swap so the image build cannot exhaust the RAM"
SWAP_MIB=$(free -m | awk '/^Swap:/ {print $2}')
if [ "${SWAP_MIB:-0}" -lt 2048 ] && [ ! -f /swapfile ]; then
    fallocate -l 2G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

echo "==> Preparing $DEPLOY_DIR and the push-to-deploy repository"
mkdir -p "$DEPLOY_DIR/data"
# The container runs as uid 10001, so the state volume must belong to it.
chown -R "$BOT_UID:$BOT_UID" "$DEPLOY_DIR/data"

if [ ! -d "$REPO_DIR" ]; then
    git init --bare --initial-branch=master "$REPO_DIR"
fi

cat > "$REPO_DIR/hooks/post-receive" <<'HOOK'
#!/usr/bin/env bash
# Checks out the pushed revision into the deploy directory and restarts the bot.
set -euo pipefail

DEPLOY_DIR=/srv/miocard
REPO_DIR=/srv/miocard.git

# git sets these for hooks; they would override the explicit flags below.
unset GIT_DIR GIT_WORK_TREE

while read -r _old new ref; do
    case "$ref" in
        refs/heads/master|refs/heads/main) ;;
        *) echo "Skipping $ref"; continue ;;
    esac

    echo "==> Deploying ${new:0:12} to $DEPLOY_DIR"
    git --git-dir="$REPO_DIR" --work-tree="$DEPLOY_DIR" checkout -f "$new"
    git --git-dir="$REPO_DIR" --work-tree="$DEPLOY_DIR" clean -fd -e data -e .env

    if [ ! -f "$DEPLOY_DIR/.env" ]; then
        echo "ERROR: $DEPLOY_DIR/.env is missing. Upload it once and push again." >&2
        exit 1
    fi

    cd "$DEPLOY_DIR"
    docker compose up -d --build
    docker image prune -f >/dev/null

    echo "==> Waiting for the container to settle"
    sleep 10
    container=$(docker compose ps -q bot)
    if [ -z "$container" ] || [ "$(docker inspect -f '{{.State.Running}}' "$container")" != "true" ]; then
        echo "ERROR: the bot container is not running. Recent log:" >&2
        docker compose logs --tail 50 bot >&2
        exit 1
    fi
    docker compose logs --tail 15 bot
    echo "==> Deployed"
done
HOOK
chmod +x "$REPO_DIR/hooks/post-receive"

echo
echo "Done. Next steps from the workstation:"
echo "  1. Copy .env to $DEPLOY_DIR/.env (scp)."
echo "  2. git remote add server ssh://root@<server-ip>$REPO_DIR"
echo "  3. git push server master"
