#!/usr/bin/env bash
# Provision the small host-runtime layer required by SGLang's native H100
# extensions. Python packages remain in the dedicated Vessl virtualenv; these
# two dependencies cannot be supplied reliably by that virtualenv because they
# are loaded by ELF/JIT tooling from the host image.
set -euo pipefail

need_numa=false
need_ninja=false
ldconfig -p 2>/dev/null | grep -q 'libnuma\.so\.1' || need_numa=true
command -v ninja >/dev/null 2>&1 || need_ninja=true

if [[ "$need_numa" == false && "$need_ninja" == false ]]; then
    echo 'sglang_runtime_dependencies=ready'
    exit 0
fi

[[ "$(id -u)" -eq 0 ]] || {
    echo 'Missing host dependencies (libnuma1 and/or ninja-build); bootstrap requires root.' >&2
    exit 2
}
command -v apt-get >/dev/null 2>&1 || {
    echo 'Missing host dependencies and apt-get is unavailable.' >&2
    exit 2
}

packages=()
[[ "$need_numa" == true ]] && packages+=(libnuma1)
[[ "$need_ninja" == true ]] && packages+=(ninja-build)
DEBIAN_FRONTEND=noninteractive apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${packages[@]}"
ldconfig
ldconfig -p 2>/dev/null | grep -q 'libnuma\.so\.1' || {
    echo 'libnuma1 installation did not expose libnuma.so.1.' >&2
    exit 1
}
command -v ninja >/dev/null 2>&1 || {
    echo 'ninja-build installation did not expose ninja.' >&2
    exit 1
}
echo 'sglang_runtime_dependencies=installed'
