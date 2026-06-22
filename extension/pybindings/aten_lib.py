"""Fallback ATen-mode pybindings shim for local CMake/pybind builds.

The OSS CMake pybind build installs `_portable_lib`, but does not currently
emit a distinct `aten_lib` extension like the BUCK build does. Tests that only
need the loader entrypoints can use the portable pybindings surface.
"""

from executorch.extension.pybindings import portable_lib as _portable_lib

# Re-export the portable loader surface, including underscore-prefixed symbols
# that `import *` would normally omit.
_create_profile_block = _portable_lib._create_profile_block
_dump_profile_results = _portable_lib._dump_profile_results
_get_operator_names = _portable_lib._get_operator_names
_get_registered_backend_names = _portable_lib._get_registered_backend_names
_is_available = _portable_lib._is_available
_load_bundled_program_from_buffer = _portable_lib._load_bundled_program_from_buffer
_load_for_executorch = _portable_lib._load_for_executorch
_load_for_executorch_from_buffer = _portable_lib._load_for_executorch_from_buffer
_load_for_executorch_from_bundled_program = (
    _portable_lib._load_for_executorch_from_bundled_program
)
_load_program = _portable_lib._load_program
_load_program_from_buffer = _portable_lib._load_program_from_buffer
_reset_profile_results = _portable_lib._reset_profile_results
_threadpool_get_thread_count = _portable_lib._threadpool_get_thread_count
_unsafe_reset_threadpool = _portable_lib._unsafe_reset_threadpool
BundledModule = _portable_lib.BundledModule
ExecuTorchMethod = _portable_lib.ExecuTorchMethod
ExecuTorchModule = _portable_lib.ExecuTorchModule
ExecuTorchProgram = _portable_lib.ExecuTorchProgram
MethodMeta = _portable_lib.MethodMeta
Verification = _portable_lib.Verification

__all__ = [
    "_create_profile_block",
    "_dump_profile_results",
    "_get_operator_names",
    "_get_registered_backend_names",
    "_is_available",
    "_load_bundled_program_from_buffer",
    "_load_for_executorch",
    "_load_for_executorch_from_buffer",
    "_load_for_executorch_from_bundled_program",
    "_load_program",
    "_load_program_from_buffer",
    "_reset_profile_results",
    "_threadpool_get_thread_count",
    "_unsafe_reset_threadpool",
    "BundledModule",
    "ExecuTorchMethod",
    "ExecuTorchModule",
    "ExecuTorchProgram",
    "MethodMeta",
    "Verification",
]
