#!/usr/bin/env bash

# An rsync remote shell, also used for ordinary Orin commands. Nothing is copied
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
    printf 'Usage: %s [USER@HOST [COMMAND ...]]\nDefault: root@10.9.71.11 through yasen@yasen-mbp\nOverride Mac: COS_DEPLOY_VIA=USER@HOST\n' "$0"
    exit 0
fi
if (($# == 0)); then
    set -- root@10.9.71.11
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
