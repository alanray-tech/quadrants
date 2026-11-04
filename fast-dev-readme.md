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

Known limits: the AMDGPU part of the checkpoint yield fix has not run on AMD hardware, and the AMDGPU graph path
(`runtime/amdgpu/graph_manager.cpp`), which splits tasks by checkpoint_id the same way, is unchanged and may have the same
problem.

## Syncing with upstream

```bash
git fetch origin                       # origin = Genesis-Embodied-AI/quadrants, fork = alanray-tech/quadrants
git push fork origin/main:main         # the fork's main only fast-forwards
git switch dev-fast
git merge origin/main                  # merge, never rebase: the history of dev-fast is not rewritten
git push fork dev-fast
```

Once upstream merges a topic, its changes arrive with `origin/main`. Upstream usually squash-merges, which can conflict
with the older version in `dev-fast`: resolve to the upstream side. Then remove the topic from the table above and delete
its branch on the fork.

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
