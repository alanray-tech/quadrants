# dev-fast

`dev-fast` is the development branch of this fork (`alanray-tech/quadrants`): the `main` of upstream
`Genesis-Embodied-AI/quadrants`, plus fixes and features that are not upstream yet, for day-to-day development (Genesis,
the GPU test suite on Windows, ...). It merges upstream often and is not used for pull requests.

The two long-lived branches of this fork:

| Branch | Role |
|---|---|
| `main` | Mirror of upstream `main`. Never committed to, only fast-forwarded |
| `dev-fast` | Upstream `main` plus the topic branches below, each merged with one `--no-ff` merge |

This file lives only on `dev-fast` and belongs to no topic: leave it out of any upstream pull request.

## Topics

Each topic has its own branch on the fork. Changes are made on the topic branch, then merged into `dev-fast`.

| Topic branch | Content | Upstream status |
|---|---|---|
| `fix/issue-750-nested-checkpoint` | Checkpoints that contain graph do-while loops, plus a one-line fix keeping the `kernel_launch_metadata` module out of `qd.__all__` | PR #944 (the one-line fix is fork-only) |
| `perf/compile-speed` | On top of the previous branch: `qd.graph.parallel_context()` inside checkpoints; faster compilation of large graph kernels (LLVM optimization level follows `external_optimization_level`); grid-stride launches clamped to resident occupancy; dynamic range bounds inlined into the worker on CUDA instead of a separate serial kernel; LLVM modules stored as bitcode in the offline cache; PTX cache keys from bitcode fingerprints; cached function source; no AST flattening of empty dataclasses | Not submitted |
| `fix/issue-1040-namedtuple-in-data-oriented` | NamedTuple members of `@qd.data_oriented` classes, iterable with `qd.static` | PR #1050 |
| `fix/svd-fp64-accuracy` | Rotation thresholds of the fp64 3x3 SVD at fp64 precision | PR #1051 |
| `fix/windows-cpu-jit-section-layout` | The CPU JIT allocates the sections of a module in one contiguous block. Fixes `IMAGE_REL_AMD64_ADDR32NB relocation requires an ordered section layout` on Windows in processes that initialized CUDA, and the crash or hang in `Program` teardown that followed it | Not submitted |
| `fix/fastcache-concurrent-access` | A fastcache entry that fails to read or write because another process holds it counts as a cache miss instead of failing the kernel call | Issue #943 |
| `fix/checkpoint-yield-after-helper-tasks` | On CPU and AMDGPU, a checkpoint with several dynamic range loops no longer yields after its first loop | Not submitted |
| `fix/windows-test-portability` | Two tests failing on Windows: the UTF-8 fixture is written with an explicit encoding, and a capfd test is skipped on Windows | Not submitted |

## Feature notes

What each topic changes for a user of `dev-fast`, what covers it, and what is known not to work. The user guide pages
named here are the ones on `dev-fast`.

### Checkpoints containing graph do-while loops (`fix/issue-750-nested-checkpoint`, PR #944)

- A `qd.checkpoint` body may contain a `qd.graph.do_while` loop; resuming past the checkpoint skips the whole loop. See
  "qd.checkpoint" in `docs/source/user_guide/graph.md`.
- Only on CUDA with compute capability 9.0 or newer and CUDA Graphs enabled. Every other backend, and CUDA with graphs
  disabled, rejects the kernel at launch.
- Inside a loop, the resume point applies only to the first iteration after `kernel.resume(...)`. Resuming into or out
  of a loop, or between sibling loops, is not supported.
- Tests: `tests/python/test_checkpoint.py`, `tests/python/quadrants/lang/fast_caching/test_src_hasher.py`.

### Graph parallel regions inside checkpoints (`perf/compile-speed`)

- `qd.graph.parallel_context()` may be used in a `checkpoints=True` kernel when the whole region is inside one explicit
  `qd.checkpoint`. All sections finish before the yield check of that checkpoint, and resume skips or replays the whole
  region. A region outside any `qd.checkpoint` in such a kernel is still a compile error. See
  "qd.graph.parallel_context" in `docs/source/user_guide/graph.md`.
- Tests: `tests/python/test_graph_parallel.py`.

### Compile speed (`perf/compile-speed`)

- `external_optimization_level` (`qd.init` option, `0` to `3`, default `3`) now also sets the LLVM optimization and
  code-generation levels (`O0` to `O3`). Lower levels cut the cold compile time of very large kernels and make them
  slower. It is part of the offline-cache key. See `docs/source/user_guide/init_options.md`.
- Large graph kernels: the pre-offload pointer CSE is skipped for kernels without field pointers (ndarray-only kernels),
  replacing a frontend statement in lowering no longer scans the whole IR for users, constant folding replaces usages
  through an index built once per fixpoint iteration instead of scanning the IR for every folded statement, and
  `bit_loop_vectorize` skips its dead-code elimination when the kernel has no quant-array loop.
- CUDA: a range-for whose bounds are read at run time used to be lowered to a serial helper kernel that computes the
  bounds, then the worker kernel. The worker now evaluates the bounds itself, which removes one kernel (one graph node)
  per such loop. Volatile bounds, and reverse-mode tasks with an adstack, keep the helper. CPU and AMDGPU are unchanged.
- CUDA: a grid-stride kernel is launched with at most one resident wave of blocks, computed from the occupancy of the
  compiled function, instead of the architecture-wide saturating grid. A smaller requested grid is kept as is.
- Offline cache: kernel LLVM modules are stored as bitcode, with a `QDBC1` prefix, instead of textual IR, and PTX cache
  keys are bitcode fingerprints (schema `ptx-cache-key-v2-bitcode`). `dev-fast` still reads the old textual entries,
  but recompiles every PTX entry once. An upstream build cannot load the new entries (it reports a parse failure), so
  do not share one offline cache directory between `dev-fast` and an upstream build.
- Python: the source text of each `qd.func` / `qd.kernel` is read and dedented once instead of at every inline site,
  and dataclass AST flattening is skipped when a kernel has no dataclass parameters.
- Tests: `tests/python/test_offload_cross.py`, `tests/python/test_runtime.py`,
  `tests/cpp/runtime/cuda/ptx_cache_test.cpp`. There is no benchmark in the repository for the compile-time gains.

### NamedTuple members of data-oriented classes (`fix/issue-1040-namedtuple-in-data-oriented`, PR #1050)

- A `@qd.data_oriented` class may hold `NamedTuple` members, iterable in kernels with `qd.static`. A NamedTuple passed
  directly as a kernel argument is still restricted. See `docs/source/user_guide/compound_types.md`.
- Tests: `tests/python/test_data_oriented_ndarray.py`, `tests/python/test_data_oriented_template_primitives.py`.

### fp64 SVD accuracy (`fix/svd-fp64-accuracy`, PR #1051)

- The fp64 3x3 SVD skipped Jacobi and QR rotations under the fp32 thresholds, which capped its reconstruction error
  near the identity at about 1e-6. It now uses thresholds for fp64, and reaches a few ulps. No API change.
- Tests: `tests/python/test_svd.py`.

### CPU JIT on Windows (`fix/windows-cpu-jit-section-layout`)

- The CPU JIT reserves the sections of each module in one block, ordered code, read-only data, read-write data. Before,
  in a process whose address space CUDA had fragmented (a pytest-xdist worker that ran CUDA tests, then CPU tests), two
  sections of one module could land more than 4 GB apart, and linking failed with `IMAGE_REL_AMD64_ADDR32NB relocation
  requires an ordered section layout`. Because the LLVM fatal-error handler throws through LLVM, the failed lookup then
  stayed registered with a dangling stack reference, and the next `Program` teardown crashed or hung forever.
- No test in the repository. It was checked with a script that fragments the address space (every module failed before,
  none after), and with 6 runs of the Genesis GPU and implicit MPM suites without any relocation error, crash or hang.
- Not fixed: the LLVM fatal-error handler still throws through LLVM, so any other LLVM fatal error can leave the process
  in the same broken state.

### Fastcache shared between processes (`fix/fastcache-concurrent-access`, issue #943)

- When another process holds or replaces a Python-side fastcache entry (WinError 5 / Errno 13 on Windows, a stale NFS
  handle on Linux), a failed read is a cache miss with a warning, a failed store is skipped, and a failed date update
  after a read is ignored. Before, the kernel call failed.
- Tests: `tests/python/quadrants/lang/fast_caching/test_python_side_cache.py`.

### Checkpoint yield with dynamic ranges (`fix/checkpoint-yield-after-helper-tasks`)

- On CPU and AMDGPU, a yielding checkpoint whose body has several range-fors with run-time bounds used to yield right
  after the first loop, skipping the others: the bound helper tasks of those loops carry no checkpoint id, and were
  taken for the end of the checkpoint. The yield check now runs after the last task of the checkpoint.
- Tests: `test_checkpoint_yields_after_its_whole_body_with_dynamic_ranges` in `tests/python/test_checkpoint.py`.
- Not checked on AMD hardware. The AMDGPU graph path (`runtime/amdgpu/graph_manager.cpp`), which splits tasks by
  checkpoint id the same way, is unchanged and may have the same problem.

### Windows test portability (`fix/windows-test-portability`)

- `test_read_file_with_arrows` writes its UTF-8 fixture with an explicit encoding, and
  `test_data_oriented_kernel_read_opaque_member_fails_fastcache` is skipped on Windows, where capfd does not capture
  stderr. Test-only change.

## Verification on Windows

Windows 11, RTX 5090, CUDA, LLVM 22.1.0. On the merged `dev-fast`, the tests of the files that the topics change, on CPU
and CUDA: 519 passed, and the 3 failures (2 tests) that `fix/windows-test-portability` then fixed. The full quadrants
test suite has not been run on `dev-fast`.

## Syncing with upstream

```bash
git fetch origin                       # origin = Genesis-Embodied-AI/quadrants, fork = alanray-tech/quadrants
git push fork origin/main:main         # the fork's main only fast-forwards
git switch dev-fast
git merge origin/main                  # merge, never rebase: the history of dev-fast is not rewritten
git push fork dev-fast
```

Once upstream merges a topic, its changes arrive with `origin/main`. Upstream usually squash-merges, which can conflict
with the older version in `dev-fast`: resolve to the upstream side. Then remove the topic from the tables above and
delete its branch on the fork.

## Adding or changing a topic

```bash
git switch -c fix/<name> origin/main   # or from the topic branch it depends on
# commit the change
git push fork fix/<name>
git switch dev-fast
git merge --no-ff fix/<name> -m "Merge fix/<name> into dev-fast"
git push fork dev-fast
```

Apart from merge commits and this file, do not commit to `dev-fast` directly, so that every topic can still become its
own pull request, or be dropped on its own. To change an existing topic, commit on its branch and merge it into
`dev-fast` again, and update its row in the table.
