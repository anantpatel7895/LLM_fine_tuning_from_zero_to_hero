#!/usr/bin/env bash
# Phase 00 follow-up: diagnose the three blockers the first hardware report found.
#
#   ./scripts/00b_diagnose_blockers.sh
#
# 1. NVML driver/library version mismatch  -> no GPU is usable until fixed
# 2. Root filesystem 92% full, 7.9 GB free -> cannot hold even one 8B model
# 3. pyenv FT_VENV does not exist          -> expected, not yet created
#
# This script only READS. It changes nothing. Paste the output back.

set -uo pipefail          # deliberately no -e: every probe should run even if one fails

hr() { printf '\n%s\n%s\n' "=== $1 ===" "$(printf '%.0s-' {1..70})"; }

echo "Blocker diagnosis — $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "host: $(hostname)"
echo "kernel: $(uname -r)"

# ---------------------------------------------------------------------------
hr "1a. Which NVIDIA kernel module is CURRENTLY LOADED"
# If this file is missing, no nvidia module is loaded at all.
cat /proc/driver/nvidia/version 2>/dev/null || echo "no nvidia kernel module loaded"

hr "1b. Which NVIDIA kernel module is INSTALLED ON DISK"
# This is the decisive comparison. If the on-disk module matches the NVML library
# version (580.178) but the loaded one does not, a REBOOT fixes it. If no 580 module
# exists on disk, the driver install is broken and a reboot will NOT help.
modinfo -F version nvidia 2>/dev/null || echo "modinfo: nvidia module not found on disk"
echo "--- all nvidia.ko copies present ---"
find /lib/modules -name 'nvidia.ko*' 2>/dev/null | head -20 || true

hr "1c. DKMS status (did the module build for this kernel?)"
dkms status 2>/dev/null || echo "dkms not installed / not in PATH"

hr "1d. Installed NVIDIA packages"
{ dpkg -l 2>/dev/null | grep -iE 'nvidia|cuda' | awk '{print $1, $2, $3}' | head -30; } \
    || echo "dpkg unavailable"

hr "1e. Loaded nvidia modules and whether anything is using them"
lsmod 2>/dev/null | grep -i nvidia || echo "no nvidia modules loaded"

hr "1f. Is a reboot already pending?"
if [ -f /var/run/reboot-required ]; then
    echo "YES — /var/run/reboot-required exists"
    cat /var/run/reboot-required.pkgs 2>/dev/null | head -10
else
    echo "no reboot-required flag"
fi

hr "1g. nvidia-smi / NVML versions as reported by the tools themselves"
nvidia-smi --version 2>&1 | head -10 || true

hr "1h. Are the GPUs even visible on the PCI bus?"
# This works regardless of driver state. If the cards are not here, the problem is
# the VM/host, not the driver.
lspci 2>/dev/null | grep -iE 'nvidia|3d controller|vga' || echo "lspci unavailable"

# ---------------------------------------------------------------------------
hr "2a. ALL filesystems (looking for a big volume to move HF_HOME onto)"
df -hT -x tmpfs -x devtmpfs -x squashfs 2>/dev/null || df -h

hr "2b. Block devices — is there an unmounted or unformatted data disk?"
lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT 2>/dev/null || echo "lsblk unavailable"

hr "2c. What is actually consuming the 90 GB on /"
du -xhd1 / 2>/dev/null | sort -rh | head -15 || true

hr "2d. Existing HuggingFace cache size, if any"
for d in "$HOME/.cache/huggingface" "${HF_HOME:-}" /opt/hf /mnt /data /scratch; do
    [ -n "$d" ] && [ -e "$d" ] && printf '%-40s %s\n' "$d" "$(du -sh "$d" 2>/dev/null | cut -f1)"
done

# ---------------------------------------------------------------------------
hr "3a. pyenv state"
command -v pyenv >/dev/null 2>&1 && {
    echo "pyenv: $(pyenv --version 2>&1)"
    echo "--- installed versions ---"
    pyenv versions 2>&1 | head -20
} || echo "pyenv not installed or not on PATH"

hr "3b. System python"
for p in python python3 python3.12; do
    command -v "$p" >/dev/null 2>&1 && printf '%-12s %s\n' "$p" "$($p --version 2>&1)"
done

hr "3c. Build deps needed to compile a CPython via pyenv"
for pkg in build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
           libsqlite3-dev libffi-dev liblzma-dev; do
    if dpkg -s "$pkg" >/dev/null 2>&1; then printf '  present  %s\n' "$pkg"
    else printf '  MISSING  %s\n' "$pkg"; fi
done

hr "done"
echo "Paste all of the above back."
echo
echo "The decisive lines are 1a vs 1b (reboot vs reinstall) and 2b (is there a"
echo "spare disk for models). Everything else is supporting detail."
