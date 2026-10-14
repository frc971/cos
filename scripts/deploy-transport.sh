#!/usr/bin/env bash

# Deploy the current checkout, or act as an rsync remote shell. Nothing is copied
# to the Mac: its SSH client carries stdin/stdout to the Orin using the Mac's key.
set -euo pipefail

via=${COS_DEPLOY_VIA:-yasen@yasen-mbp}
# Allocate terminals for an interactive shell, but keep rsync's stream binary.
terminal=-T
if [[ -t 0 && -t 1 ]]; then
    terminal=-t
fi
ssh_options=(
    "$terminal" -a
    -o BatchMode=yes
    -o PreferredAuthentications=publickey
    -o StrictHostKeyChecking=yes
    -o ConnectTimeout=10
    -o ServerAliveInterval=15
    -o ServerAliveCountMax=3
)

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    printf 'Usage: %s [--vision | USER@HOST [COMMAND ...]]\nBuild and deploy the current checkout (run from its root).\nDefault: root@10.9.71.11 through yasen@yasen-mbp\nVision: root@10.89.71.11\nExplicit SSH arguments select transport mode for commands or rsync.\nOverride Mac: COS_DEPLOY_VIA=USER@HOST\n' "$0"
    exit 0
fi
if (($# == 0)) || [[ ${1:-} == --vision ]]; then
    if (($# > 1)); then
        printf 'Deployment accepts only --vision; use USER@HOST for remote commands.\n' >&2
        exit 2
    fi
    if [[ ! -f CMakeLists.txt || ! -x ./scripts/build.sh ]]; then
        printf 'Run deployment from the checkout root containing ./scripts/build.sh.\n' >&2
        exit 2
    fi

    destination=root@10.9.71.11
    if [[ ${1:-} == --vision ]]; then
        destination=root@10.89.71.11
    fi
    transport="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")"
    # rsync parses -e itself; double quotes preserve spaces in the script path.
    rsync_shell="bash \"${transport//\"/\"\"}\""

    ./scripts/build.sh
    printf 'Deploying %s to %s through %s\n' "$PWD" "$destination" "$via"
    "$transport" "$destination" 'mkdir -p /root'
    for directory in tests main examples tools lib; do
        rsync -avz --delete -e "$rsync_shell" "build/$directory/" "$destination:/root/$directory/"
    done
    rsync -avz --delete -e "$rsync_shell" constants/ "$destination:/root/constants/"
    rsync -avz -e "$rsync_shell" systemd/ "$destination:/etc/systemd/system/"
    exit 0
fi

# SSH sends its remote command through the Mac's login shell. Quote each inner
# SSH argument for that shell; preserve the original Orin command arguments.
# POSIX single-quote escaping works with the Mac's usual zsh and bash shells.
remote_command='exec /usr/bin/ssh'
for argument in "${ssh_options[@]}" "$@"; do
    quoted_argument=${argument//\'/\'\\\'\'}
    remote_command+=" '$quoted_argument'"
done

# rsync supplies "-l USER HOST ...", which the inner SSH accepts directly.
exec ssh "${ssh_options[@]}" "$via" "$remote_command"
