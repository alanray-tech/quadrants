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
| `perf/compile-speed` | On top of the previous branch: `qd.graph.parallel_context()` inside checkpoints; faster compilation of large graph kernels (LLVM optimization level follows `external_optimization_level`); grid-stride launches clamped to resident occupancy; dynamic range bounds inlined into the worker on CUDA instead of a separate serial kernel; LLVM modules stored as bitcode in the offline cache; PTX cache keys from bitcode fingerprints; cached function source; no AST flattening of empty dataclasses; whole-kernel CSE with a per-round use index (upstream PR #1013, cherry-picked) | Not submitted (the CSE part is PR #1013) |
| `fix/issue-1040-namedtuple-in-data-oriented` | NamedTuple members of `@qd.data_oriented` classes, iterable with `qd.static` | PR #1050 |
| `fix/svd-fp64-accuracy` | Rotation thresholds of the fp64 3x3 SVD at fp64 precision | PR #1051 |
| `fix/windows-cpu-jit-section-layout` | The CPU JIT allocates the sections of a module in one contiguous block. Fixes `IMAGE_REL_AMD64_ADDR32NB relocation requires an ordered section layout` on Windows in processes that initialized CUDA, and the crash or hang in `Program` teardown that followed it | Not submitted |
| `fix/fastcache-concurrent-access` | Processes sharing an offline cache directory no longer conflict on fastcache entries on Windows, and a read or write that still fails counts as a cache miss instead of failing the kernel call | Issue #943 |
| `fix/checkpoint-yield-after-helper-tasks` | On CPU and AMDGPU, a checkpoint with several dynamic range loops no longer yields after its first loop | Not submitted |
| `fix/windows-test-portability` | Three tests failing on Windows: the UTF-8 fixture is written with an explicit encoding, and two capfd tests are skipped on Windows | Not submitted |
| `fix/graph-parallel-serial-section-tasks` | The right-hand side of a bare store in a `qd.graph.parallel()` section is computed in that section's task, not in the previous section's, which raced on the CUDA graph path | Not submitted |
| `perf/checkpoint-implicit-grouping` | In a `checkpoints=True` kernel, each run of consecutive loops outside explicit checkpoints is one implicit checkpoint, so one gate kernel and one CUDA IF node, instead of one per loop | Not submitted |
| `fix/cpp-tests-without-vulkan` | The C++ tests link the SPIR-V codegen themselves when Vulkan and Metal are off, so they build in a Vulkan-off build | Not submitted |
| `perf/checkpoint-task-regions` | On top of `perf/compile-speed` and `perf/checkpoint-implicit-grouping`: every task of a `checkpoints=True` kernel belongs to a checkpoint (inlined funcs, static ifs and bare statements included), one implicit checkpoint per maximal run outside explicit ones; bare statements and func calls allowed in checkpoint bodies; `qd.CHECKPOINT_TASK_REGIONS` | Not submitted (fixes issue #956) |
| `perf/checkpoint-free-loops` | On top of `perf/checkpoint-task-regions`: a `qd.graph.do_while` holding no checkpoint joins the implicit checkpoint around it (one gate around the loop, not one per pass) and uses the plain condition kernel | Not submitted |
| `fix/checkpoint-resume-nested-loop` | On CPU and Vulkan/Metal, the work after a nested `qd.graph.do_while` that precedes the resumed checkpoint no longer runs again in the resumed pass | Not submitted |
| `perf/cuda-range-rematerialize` | On top of `perf/compile-speed`: on CUDA, scalar range bounds computed in an earlier task, and the loop body's uses of such scalars, are recomputed in the worker instead of passed through a serial task; a bound that reads device memory the loop may write keeps the serial task | Not submitted (follows PR #1024) |

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
- On CUDA, level `0` runs the `O1` IR pipeline with `O0` code generation. With the `O0` IR pipeline, workers called
  their loop bodies through function pointers, the PTX JIT charged every worker for the shared arrays of the whole
  module, and Genesis' `kernel_radix_sort` (192 KB of shared arrays over its tasks) failed to load with
  `CUDA_ERROR_INVALID_PTX`. Test: `test_static_shared_arrays_of_many_tasks_without_llvm_optimization`.
- CUDA: a PTX that fails to load reports the JIT error log (the ptxas messages) with `CUDA_ERROR_INVALID_PTX`.
- Large graph kernels: whole-kernel CSE rewrites the users of an eliminated statement through a use index built once
  per round, and erases a round's eliminated statements in one pass per block (the 3 commits of upstream PR #1013 by
  Roushelfy, cherry-picked), so the pre-offload pointer CSE is linear instead of quadratic in the kernel size. It runs
  on every kernel again, as upstream: an earlier early return for kernels without field pointers is dropped (on the
  PR19 Genesis scene it saved 0.4 s of a 40-55 s build, and changed 42 of 72 PTX modules away from upstream).
  Replacing a frontend statement in lowering no longer scans the whole IR for users, constant folding replaces usages
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
  `tests/cpp/runtime/cuda/ptx_cache_test.cpp`, `tests/cpp/transforms/whole_kernel_cse_test.cpp`. There is no benchmark
  in the repository for the compile-time gains. The C++ tests need `QD_LIB_DIR` set to
  `<quadrants package>/_lib/runtime` (every test that creates a `Program` throws without it).

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

- On Windows, the builtin `open` and `os.utime` open a file without `FILE_SHARE_DELETE`, so `os.replace` onto a
  fastcache entry that another process reads or touches failed with WinError 5, and opening an entry while it was
  replaced failed with Errno 13. Entries are now read and touched through handles shared for deletion
  (`_fast_caching/_win_shared_file.py`) and replaced with a POSIX-semantics rename, which swaps the name while other
  handles are open; those keep reading the previous content. File systems without POSIX rename support fall back to
  `os.replace`.
- On every platform, a read that still fails (or a stale NFS handle on Linux) is a cache miss with a warning, a store
  that still fails is skipped, and a failed date update after a read is ignored. Before, the kernel call failed.
- Four processes storing and loading one key for 8 s: 7433 failed replaces, 1842 failed reads and 68 failed touches
  with the builtin calls, none with the shared handles.
- An entry another process reads with an older quadrants (builtin `open`, no `FILE_SHARE_DELETE`) still blocks the
  replace: that store is skipped.
- Tests: `tests/python/quadrants/lang/fast_caching/test_python_side_cache.py`, including a Windows-only one that
  replaces and reads an entry while a shared handle holds it open.

### Checkpoint yield with dynamic ranges (`fix/checkpoint-yield-after-helper-tasks`)

- On CPU and AMDGPU, a yielding checkpoint whose body has several range-fors with run-time bounds used to yield right
  after the first loop, skipping the others: the bound helper tasks of those loops carry no checkpoint id, and were
  taken for the end of the checkpoint. The yield check now runs after the last task of the checkpoint.
- Tests: `test_checkpoint_yields_after_its_whole_body_with_dynamic_ranges` in `tests/python/test_checkpoint.py`.
- Not checked on AMD hardware. The AMDGPU graph path (`runtime/amdgpu/graph_manager.cpp`), which splits tasks by
  checkpoint id the same way, is unchanged and may have the same problem.

### Implicit checkpoint grouping (`perf/checkpoint-implicit-grouping`)

- In a `checkpoints=True` kernel, every top-level for-loop outside an explicit `qd.checkpoint` used to become its own
  implicit checkpoint, which costs one gate kernel and one IF conditional node on the CUDA graph path (SM 9.0+). Now
  each run of consecutive top-level for-loops, with the `qd.loop_config(...)` calls between them, is one implicit
  checkpoint. Bare statements, explicit checkpoints and `qd.graph.do_while` loops end a run.
- Resume and yield behave as before: an implicit checkpoint never yields and cannot be a resume target, so the loops
  of one run were always skipped or run together.
- Kernels that relied on the number of implicit slots see fewer entries in `checkpoint_user_labels_by_cp_id`, and
  fewer checkpoints in `get_graph_num_checkpoints_on_last_call()`.
- A graph kernel of 256 small loops with one explicit checkpoint on an RTX 5090: 772 graph nodes and 4.1 ms per
  launch before, 262 nodes and 0.68 ms after (0.48 ms without checkpoints). A kernel with checkpoints still costs about
  0.13 ms more per launch than one without, independently of its size; that is not investigated yet.
- Same idea as the continuation regions of the `cuda-graph-qipc` pipeline builder, where one gate and one IF cover a
  maximal run of ordinary work.
- Tests: the `test_autowrap_*` and `test_grouped_implicit_loops_*` tests in `tests/python/test_checkpoint.py`.

### Checkpoint regions at the task level (`perf/checkpoint-task-regions`)

- Before, only loops written in a `qd.checkpoint` body, and runs of top-level loops (implicit checkpoints), had a
  checkpoint id. The loops of an inlined `@qd.func`, static-if bodies and bare stores outside explicit checkpoints ran
  with cp_id -1: a resume ran them again and a yield did not stop them (issue #956). A one-kernel solve made of funcs,
  as Genesis' Newton solve, needed an explicit never-yield checkpoint around every func call.
- Now the ASTBuilder opens the implicit checkpoints: in a `checkpoints=True` kernel, a statement inserted into the root
  block while no checkpoint is open opens one, closed by the next explicit checkpoint or `qd.graph.do_while` boundary.
  Graph do-while bodies, static ifs and `requires_top_level` funcs all insert into the root block, so each maximal run
  of tasks, whatever their source, is one implicit checkpoint (one gate kernel and one IF node on the CUDA graph path).
  Replaces the AST grouping of `perf/checkpoint-implicit-grouping`, which had to stop at bare statements because a
  `with` body opens a variable scope. Graph manager and launchers are unchanged.
- Relaxed: bare statements and bare func calls in a checkpoint body (no `if qd.static(True):` needed), and a
  `qd.graph.parallel_context()` outside explicit checkpoints. The offloader closes a serial bucket before a pure
  statement of another checkpoint, so a later checkpoint's bare store does not read a value computed in a task that a
  resume skipped. A pure statement computed in an implicit checkpoint next to a bare store, and used by a later
  checkpoint, is still skipped with that store on resume.
- `qd.CHECKPOINT_TASK_REGIONS` (`True`) tells this behaviour apart from builds without it; read it with
  `getattr(qd, "CHECKPOINT_TASK_REGIONS", False)`.
- Probes of the IPC perf member (6-pass do-while, yield once at pass 3, expected [prologue 1, before 6, body 7,
  after 6]), CPU and CUDA, before -> after: prologue as a func [2, 6, 7, 6] -> [1, 6, 7, 6]; section before as a
  func [1, 7, 7, 6] -> [1, 6, 7, 6]; body as a func under `if qd.static(True)` [1, 6, 7, 6] (already right); every
  section a func [2, 6, 6, 6] -> [1, 6, 7, 6].
- RTX 5090, exclusive GPU, a 1000-pass do-while holding 20 tiny tasks: written as loops or as func calls, 28 graph
  nodes, 2 checkpoints, 43.8 us per pass; each func call in an explicit never-yield checkpoint (the Genesis-newton-v1
  workaround), 88 nodes, 22 checkpoints, 180.2 us per pass. So one checkpoint costs about 6.8 us per pass (gate kernel
  + IF node), unchanged by this topic; what changes is that funcs no longer need the workaround.
- A checkpoint body holding a `qd.graph.do_while` (PR #944) takes the CUDA graph path on the RTX 5090, before and
  after: 19 graph nodes, 4 checkpoints per launch. Quadrants 1.4.0 (without PR #944) logs "not yet supported on the
  CUDA graph path; falling back to the non-graph launch" for it, and misses the yield.
- Tests: `test_sections_as_funcs_*`, `test_yield_skips_following_top_level_func`, `test_static_if_and_bare_*`,
  `test_checkpoint_bare_store_belongs_to_checkpoint`, `test_checkpoint_body_may_be_a_bare_func_call`,
  `test_capability_flag` in `tests/python/test_checkpoint.py`, and
  `test_graph_parallel_context_outside_checkpoint_joins_implicit_checkpoint` in `tests/python/test_graph_parallel.py`.

### Checkpoint-free loops (`perf/checkpoint-free-loops`)

- With task-level regions, every `qd.graph.do_while` was a boundary of the implicit checkpoints, so a loop holding
  none (a PCG loop) launched a gate kernel and an IF node on every pass, and its condition kernel was the with-yield
  variant. The IPC perf member measured, on the mono Newton kernel (mpm_drop, nsys): 1755 gates per step (1.05 ms),
  PCG kernels 10-20% slower in the gated region, and the with-yield condition kernel 1.70 vs 1.27 ms per step.
- Now a `qd.graph.do_while` whose body holds no `with qd.checkpoint` joins the implicit checkpoint open at its
  position, which owns it whole (as a loop inside an explicit checkpoint, PR #944): one gate and one IF around the
  WHILE node per pass of the enclosing level. No resume can land inside such a loop. The graph manager picks the
  condition kernel per level: with-yield only for a level holding a yielding checkpoint. Loops holding a checkpoint
  stay boundaries, and the regions after the last explicit checkpoint stay gated (a yield must stop them).
- CPU, AMDGPU, GFX and the flat checkpoint graph cannot skip a whole loop: they keep rejecting a loop owned by an
  explicit checkpoint, and gate the tasks of a loop owned by an implicit one one by one (the loop exited with its
  flag at 0, so it ends after one pass; `fix/checkpoint-resume-nested-loop` keeps the resume point there).
- RTX 5090, exclusive GPU, an outer do-while of 1000 passes holding a task, a checkpoint-free inner do-while of 10
  passes with 2 tiny tasks, and an explicit never-yield checkpoint: 22 graph nodes, 5 checkpoints, 149.7 us per outer
  pass before; 20 nodes, 4 checkpoints, 91.0 us after (-39%). Gates per outer pass: 13 before (one per inner pass), 3
  after (prefix with the loop, the explicit checkpoint, the suffix). A checkpoint-free do-while of 20 tiny tasks:
  43.5 -> 38.0 us per pass (its single region now holds the loop, one gate per launch).
- Tests: `test_checkpoint_free_nested_loop_joins_the_region_around_it` (counts, checkpoint ids, the loop's owner,
  4 IF nodes) and `test_checkpoint_free_nested_loop_on_flat_checkpoint_graph` in
  `tests/python/test_checkpoint.py`.

### Resume past a nested loop (`fix/checkpoint-resume-nested-loop`)

- `kernel.resume(from_checkpoint=cp)` skips the checkpoints before `cp` on the first pass of the loop that holds `cp`.
  The CPU and GFX host drivers dropped the resume point at every loop level's condition check, also when the level
  ended. A nested `qd.graph.do_while` before the resumed checkpoint has its body skipped in the resumed pass and its
  flag already 0, so it ends after one pass: the resume point was dropped there and the work after the nested loop
  ran again (in Genesis' one-kernel Newton solve, `func_start_line_search` after the PCG loop on a SEARCH resume).
- Now CPU and GFX reset it only when a level runs another pass, as the CUDA cond-with-yield kernel and the AMDGPU
  driver already did. Based on upstream `b2f658e58`; upstream main has the same code.
- The IPC perf member's repro (outer do-while: a func with a task, a nested 3-pass do-while and a task; a checkpoint
  yielding on pass 3 of 6; a func), counts [prologue, head, inner, tail, body, after], expected [1, 6, 18, 6, 7, 6]:
  dev-fast `db6bdff96` gave tail 7 on CPU and was right on CUDA. The new test failed on CPU before the fix and passes
  on CPU and CUDA after it, on the topic branch alone (172 checkpoint/graph tests) and on `dev-fast`.
- Tests: `test_resume_skips_work_after_nested_loop_on_first_pass` in `tests/python/test_checkpoint.py` (GFX not run:
  Vulkan is off in these builds).

### Scalar range bounds recomputed in the worker (`perf/cuda-range-rematerialize`)

- CUDA only, and only in kernels without an adstack. Integer arithmetic over scalar arguments, constants and ndarray
  shapes is recomputed by the task that uses it instead of being written to a global temporary by a serial task. This
  covers range bounds computed in an earlier task (in Genesis' contact0, a reset loop runs between the bounds and the
  loop that uses them) and the loop body's own uses of those scalars. Single-store locals are forwarded right before
  offloading, so `n = x.shape[0]` is seen through when CFG optimization is disabled, as in Genesis.
- A bound inlined by `perf/compile-speed` that loads device memory now keeps the serial task when the loop body may
  write that memory (store, atomic, function call). Every worker reads an inlined bound when it enters the grid-stride
  loop, so a loop that decrements or clears its own count ran a different number of iterations.
- Genesis contact0 (hughperkins' benchmark from issue #929, RTX 5090, exclusive GPU, 100 graph calls): 3 tasks become
  2, and the GPU span per call drops from 25.73 to 24.57 us at 64 environments, 32.86 to 32.22 us at 1024 and
  77.21 to 76.43 us at 8192, with bit-identical contact queues. The gain is the removed 0.9 us serial task and its
  dependency.
- Same approach as PR #1024 for the scalar part. Unlike #1024, bounds that load device scalars are also inlined
  (from `perf/compile-speed`).
- Not checked: whether the write check sends any range bound of the Newton v1 PCG back to a serial task.
- Tests: the `test_cuda_*` tests in `tests/python/test_offload_cross.py`.

### Serial stores in graph parallel sections (`fix/graph-parallel-serial-section-tasks`)

- The offloader splits buffered serial statements into tasks by the region of their side effects only, so the pure
  statements computing the right-hand side of a bare store in a `qd.graph.parallel()` section, which come before the
  store, were computed in the previous section's task and handed to the store through a global temporary. On the CUDA
  graph path those two tasks can run concurrently: the store read the temporary before it was written (zero on the first
  launch of the graph, the previous launch's value later). The same happened to a bare store right after a region.
- Now a section's task is closed before a pure statement from elsewhere, so each store computes its value in its own
  task. Pure-only tasks keep their previous region and their always-run checkpoint id.
- Before the fix, `test_graph_parallel_back_to_back_regions_serial_sections_keep_join` failed in 25 of 30 fresh
  processes on upstream `main`; after it, in none.
- Tests: `test_graph_parallel_serial_section_computes_its_own_store_value` in `tests/python/test_graph_parallel.py`,
  which checks the offloaded IR for global temporaries and so fails deterministically without the fix.
- Not covered: a pure value that CSE shares between two sections is still computed in the first one and read by the
  other through a global temporary.

### C++ tests without Vulkan (`fix/cpp-tests-without-vulkan`)

- The SPIR-V codegen is always built, but `quadrants_core` only links it for the Vulkan and Metal backends: with both
  off, `quadrants_cpp_tests` failed to link the shader and SPIR-V version tests of `tests/cpp/codegen`. The tests now
  link it themselves in that case. Build-only change; builds with Vulkan or Metal are unchanged.
- To build and run the C++ tests: `-DQD_BUILD_TESTS:BOOL=ON` in `CMAKE_ARGS`, then `build/quadrants_cpp_tests.exe` with
  `QD_LIB_DIR` set to `<quadrants package>/_lib/runtime`. With Vulkan off on Windows: 186 tests, all passing.

### Windows test portability (`fix/windows-test-portability`)

- `test_read_file_with_arrows` writes its UTF-8 fixture with an explicit encoding, and
  `test_data_oriented_kernel_read_opaque_member_fails_fastcache` and `test_loop_config_name_nested_loop_warns` are
  skipped on Windows, where capfd does not capture stderr. Test-only change.

## Verification on Windows

Windows 11, RTX 5090, CUDA, LLVM 22.1.0. On the merged `dev-fast`, the test files that the topics change plus the
graph, checkpoint, offload and autodiff test files, on CPU and CUDA: 708 passed, 19 skipped, 2 xfailed, none failed.
The three back-to-back `qd.graph.parallel_context()` tests passed 30 runs each in fresh processes. The full quadrants
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
