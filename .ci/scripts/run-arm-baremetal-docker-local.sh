#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

script_dir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
repo_root=$(cd "${script_dir}/../.." && pwd)

test_suite="${1:-test_pytest_ops_tosa}"
image_tag="${EXECUTORCH_ARM_DOCKER_IMAGE_TAG:-executorch-local-arm-sdk:latest}"
sccache_disable="${SCCACHE_DISABLE:-1}"
skip_pytorch_image_build="${EXECUTORCH_ARM_DOCKER_SKIP_PYTORCH:-1}"
use_pt_pinned_commit="${EXECUTORCH_ARM_DOCKER_USE_PT_PINNED_COMMIT:-0}"
install_executorch_args="${EXECUTORCH_ARM_DOCKER_INSTALL_EXECUTORCH_ARGS:---minimal --editable}"
build_docs="${EXECUTORCH_ARM_DOCKER_BUILD_DOCS:-0}"
conda_env_name="${EXECUTORCH_ARM_DOCKER_CONDA_ENV:-py_3.10}"
conda_env_volume="${EXECUTORCH_ARM_DOCKER_CONDA_ENV_VOLUME:-executorch-arm-conda-${conda_env_name//[^[:alnum:]_.-]/_}}"
persist_conda_env="${EXECUTORCH_ARM_DOCKER_PERSIST_CONDA_ENV:-1}"
force_reinstall="${EXECUTORCH_ARM_DOCKER_FORCE_REINSTALL:-0}"

if ! docker image inspect "${image_tag}" >/dev/null 2>&1; then
  echo "[arm-docker-local] building ${image_tag}"
  (
    cd "${repo_root}/.ci/docker"
    build_args=()
    if [[ "${skip_pytorch_image_build}" != "0" ]]; then
      build_args+=(--build-arg "SKIP_PYTORCH=yes")
    fi
    BUILD_DOCS="${build_docs}" \
    SCCACHE_DISABLE="${sccache_disable}" \
      ./build.sh "ci-image:executorch-ubuntu-22.04-arm-sdk" \
      "${build_args[@]}" \
      -t "${image_tag}"
  )
fi

echo "[arm-docker-local] running ${test_suite} in ${image_tag}"

docker_args=(
  --rm
  -t
  -e ARM_FVP_INSTALL_I_AGREE_TO_THE_CONTAINED_EULA="${ARM_FVP_INSTALL_I_AGREE_TO_THE_CONTAINED_EULA:-True}"
  -e ARM_TEST_ENABLE_DIRECT_BACKEND_LOWERING="${ARM_TEST_ENABLE_DIRECT_BACKEND_LOWERING:-0}"
  -e SCCACHE_DISABLE="${sccache_disable}"
  -e EXECUTORCH_ARM_DOCKER_USE_PT_PINNED_COMMIT="${use_pt_pinned_commit}"
  -e EXECUTORCH_ARM_DOCKER_INSTALL_EXECUTORCH_ARGS="${install_executorch_args}"
  -e EXECUTORCH_ARM_DOCKER_CONDA_ENV="${conda_env_name}"
  -e EXECUTORCH_ARM_DOCKER_FORCE_REINSTALL="${force_reinstall}"
)
if [[ "${persist_conda_env}" != "0" ]]; then
  docker_args+=(-v "${conda_env_volume}:/opt/conda/envs/${conda_env_name}")
fi
docker_args+=(
  -v "${repo_root}:/work/executorch"
  -w /work/executorch
  "${image_tag}"
  bash -lc '
    set -euo pipefail
    source /opt/conda/etc/profile.d/conda.sh
    conda activate "${EXECUTORCH_ARM_DOCKER_CONDA_ENV}"
    source .ci/scripts/utils.sh
    install_marker=".ci/local/arm-docker-install-ready-${EXECUTORCH_ARM_DOCKER_CONDA_ENV}"
    setup_marker="examples/arm/arm-scratch/setup_path.sh"
    install_args=()
    if [[ "${EXECUTORCH_ARM_DOCKER_USE_PT_PINNED_COMMIT:-0}" != "0" ]]; then
      install_args+=(--use-pt-pinned-commit)
    fi
    if [[ -n "${EXECUTORCH_ARM_DOCKER_INSTALL_EXECUTORCH_ARGS:-}" ]]; then
      # shellcheck disable=SC2206
      extra_install_args=(${EXECUTORCH_ARM_DOCKER_INSTALL_EXECUTORCH_ARGS})
      install_args+=("${extra_install_args[@]}")
    fi
    if [[ "${EXECUTORCH_ARM_DOCKER_FORCE_REINSTALL:-0}" != "0" || ! -f "${install_marker}" ]]; then
      install_executorch "${install_args[@]}"
      python -m pip install pytest-xdist
      mkdir -p "$(dirname "${install_marker}")"
      touch "${install_marker}"
    fi
    if [[ "${EXECUTORCH_ARM_DOCKER_FORCE_REINSTALL:-0}" != "0" || ! -f "${setup_marker}" ]]; then
      .ci/scripts/setup-arm-baremetal-tools.sh
    fi
    backends/arm/test/test_arm_baremetal.sh '"${test_suite}"'
  '
)

docker run "${docker_args[@]}"
