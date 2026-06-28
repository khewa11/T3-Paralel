#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./build.sh [options] [target]

Options:
  -a, --arch ARCH        CUDA arch flag, e.g. sm_75, sm_86, sm_89
  -c, --conda PREFIX     Conda prefix to use for headers/libs and host compiler
  -u, --cuda-home PATH   CUDA installation prefix
  -h, --help            Show this help

Examples:
  ./build.sh
  ./build.sh --arch sm_89
  ./build.sh --arch sm_89 bench2
  ./build.sh --conda "$CONDA_PREFIX" --cuda-home /usr/local/cuda-12.4 all
EOF
}

arch="${ARCH:-sm_75}"
target="all"
conda_prefix="${CONDA_PREFIX:-}"
cuda_home="${CUDA_HOME:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -a|--arch)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
      arch="$2"
      shift 2
      ;;
    -c|--conda)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
      conda_prefix="$2"
      shift 2
      ;;
    -u|--cuda-home)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
      cuda_home="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      target="$1"
      shift
      ;;
  esac
done

if [[ -z "$conda_prefix" ]] && command -v conda >/dev/null 2>&1; then
  conda_prefix="$(conda info --base 2>/dev/null || true)"
fi

if [[ -n "$conda_prefix" ]]; then
  host_compiler_dir="$conda_prefix/bin"
else
  host_compiler_dir=""
fi

if [[ -z "$cuda_home" ]]; then
  if [[ -n "${CUDA_HOME:-}" ]]; then
    cuda_home="$CUDA_HOME"
  elif command -v nvcc >/dev/null 2>&1; then
    nvcc_path="$(command -v nvcc)"
    cuda_home="$(cd "$(dirname "$nvcc_path")/.." && pwd)"
  elif [[ -d /usr/local/cuda ]]; then
    cuda_home="/usr/local/cuda"
  fi
fi

make_args=(ARCH="-arch=${arch}")

if [[ -n "$conda_prefix" ]]; then
  make_args+=(CONDA_PREFIX="$conda_prefix")
fi

if [[ -n "$host_compiler_dir" ]]; then
  make_args+=(HOST_COMPILER_DIR="$host_compiler_dir")
fi

if [[ -n "$cuda_home" ]]; then
  make_args+=(CUDA_HOME="$cuda_home")
fi

exec make "${make_args[@]}" "$target"
