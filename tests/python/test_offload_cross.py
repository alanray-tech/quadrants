import numpy as np

import quadrants as qd
from quadrants.lang import impl

from tests.helpers import test_utils


@test_utils.test()
def test_offload_with_cross_block_locals():
    ret = qd.field(qd.f32)

    qd.root.place(ret)

    @qd.kernel
    def ker():
        s = 0
        for i in range(10):
            s += i
        ret[()] = s

    ker()

    assert ret[()] == 45


@test_utils.test()
def test_offload_with_cross_block_locals2():
    ret = qd.field(qd.f32)

    qd.root.place(ret)

    @qd.kernel
    def ker():
        s = 0
        for i in range(10):
            s += i
        ret[()] = s
        s = ret[()] * 2
        for i in range(10):
            qd.atomic_add(ret[()], s)

    ker()

    assert ret[()] == 45 * 21


@test_utils.test()
def test_offload_with_cross_block_locals3():
    ret = qd.field(qd.f32, shape=())

    @qd.kernel
    def ker():
        s = 1
        t = s
        for i in range(10):
            s += i
        ret[()] = t

    ker()

    assert ret[()] == 1


@test_utils.test()
def test_offload_with_cross_block_locals4():
    ret = qd.field(qd.f32, shape=())

    @qd.kernel
    def ker():
        a = 1
        b = 0
        for i in range(10):
            b += a
        ret[()] = b

    ker()

    assert ret[()] == 10


@test_utils.test()
def test_offload_with_flexible_bounds():
    s = qd.field(qd.i32, shape=())
    lower = qd.field(qd.i32, shape=())
    upper = qd.field(qd.i32, shape=())

    @qd.kernel
    def ker():
        for i in range(lower[()], upper[()]):
            s[()] += i

    lower[()] = 10
    upper[()] = 20
    ker()

    assert s[()] == 29 * 10 // 2


@test_utils.test(arch=qd.cuda, offline_cache=False)
def test_cuda_dynamic_range_bounds_are_inlined_into_worker_kernel():
    begin = qd.ndarray(qd.i32, shape=())
    end = qd.ndarray(qd.i32, shape=())
    out = qd.ndarray(qd.i32, shape=(32,))

    @qd.kernel(graph=True)
    def fill_dynamic_interval(
        begin: qd.types.ndarray(qd.i32, 0),
        end: qd.types.ndarray(qd.i32, 0),
        out: qd.types.ndarray(qd.i32, 1),
    ):
        for i in range(begin[()] + 1, end[()] - 2):
            out[i] += i + 1

    for lower, upper in ((2, 13), (7, 24)):
        begin.from_numpy(np.array(lower, dtype=np.int32))
        end.from_numpy(np.array(upper, dtype=np.int32))
        out.from_numpy(np.zeros(32, dtype=np.int32))
        fill_dynamic_interval(begin, end, out)

        # A dynamic range previously lowered to one scalar serial helper plus
        # the worker range task. CUDA now reads both live device bounds in the
        # worker kernel, so the graph contains exactly that one task/node.
        assert impl.get_runtime().prog.get_num_offloaded_tasks_on_last_call() == 1
        assert impl.get_runtime().prog.get_graph_num_nodes_on_last_call() == 1

        expected = np.zeros(32, dtype=np.int32)
        for i in range(lower + 1, upper - 2):
            expected[i] = i + 1
        np.testing.assert_array_equal(out.to_numpy(), expected)


@test_utils.test(arch=qd.cuda, offline_cache=False)
def test_cuda_scalar_bounds_and_loop_uses_are_recomputed_in_worker_kernel():
    out = qd.field(qd.i32, shape=128)

    @qd.kernel
    def scalar_bounds(begin: qd.i32, chunks: qd.i32):
        for i in range(begin, 4 * chunks):
            out[i + 16] = i // chunks + i % chunks

    for begin, chunks in ((-7, 5), (3, 9), (7, 1), (0, 0)):
        out.fill(-123)
        scalar_bounds(begin, chunks)
        assert impl.get_runtime().prog.get_num_offloaded_tasks_on_last_call() == 1
        expected = np.full(128, -123, dtype=np.int32)
        for i in range(begin, 4 * chunks):
            expected[i + 16] = i // chunks + i % chunks
        np.testing.assert_array_equal(out.to_numpy(), expected)


@test_utils.test(arch=qd.cuda, offline_cache=False, cfg_optimization=False)
def test_cuda_named_shape_bounds_are_recomputed_across_an_earlier_task():
    # The shape of Genesis' contact0: named shape locals, a reset loop launched before the chunked loop, and loop
    # uses of the chunk count and of the pair capacity.
    @qd.kernel(graph=True)
    def chunked_sum(
        counts: qd.types.ndarray(qd.i32, 1),
        pairs: qd.types.ndarray(qd.i32, 2),
        total: qd.types.ndarray(qd.i32, 0),
    ):
        n_envs = counts.shape[0]
        n_chunks = (24064 + n_envs - 1) // n_envs
        max_pairs = pairs.shape[0]
        for _ in range(1):
            total[()] = 0
        for flat in range(n_envs * n_chunks):
            i_b = flat // n_chunks
            chunk = flat % n_chunks
            n_pairs = qd.min(counts[i_b], max_pairs)
            for k in range(chunk * n_pairs // n_chunks, (chunk + 1) * n_pairs // n_chunks):
                qd.atomic_add(total[()], pairs[k, i_b])

    rng = np.random.default_rng(0)
    for n_envs in (17, 1024):
        counts_np = rng.integers(0, 12, size=n_envs, dtype=np.int32)
        pairs_np = rng.integers(-50, 50, size=(8, n_envs), dtype=np.int32)
        counts = qd.ndarray(qd.i32, shape=n_envs)
        pairs = qd.ndarray(qd.i32, shape=(8, n_envs))
        total = qd.ndarray(qd.i32, shape=())
        counts.from_numpy(counts_np)
        pairs.from_numpy(pairs_np)
        chunked_sum(counts, pairs, total)
        assert impl.get_runtime().prog.get_num_offloaded_tasks_on_last_call() == 2
        expected = sum(int(pairs_np[: min(c, 8), b].sum()) for b, c in enumerate(counts_np))
        assert total.to_numpy() == expected


@test_utils.test(arch=qd.cuda, offline_cache=False)
def test_cuda_device_bound_written_by_its_loop_is_read_once_before_the_launch():
    n = 1 << 20
    count = qd.ndarray(qd.i32, shape=())
    hits = qd.ndarray(qd.i32, shape=n)
    flag = qd.field(qd.i32, shape=())
    marks = qd.field(qd.i32, shape=n)

    @qd.kernel(graph=True)
    def drain(count: qd.types.ndarray(qd.i32, 0), hits: qd.types.ndarray(qd.i32, 1)):
        for i in range(count[()]):
            hits[i] = 1
            qd.atomic_sub(count[()], 1)

    @qd.kernel
    def clear_flag():
        for i in range(flag[None]):
            if i == 0:
                flag[None] = 0
            marks[i] = 1

    count.from_numpy(np.array(n, dtype=np.int32))
    hits.from_numpy(np.zeros(n, dtype=np.int32))
    drain(count, hits)
    assert impl.get_runtime().prog.get_num_offloaded_tasks_on_last_call() == 2
    assert count.to_numpy() == 0
    assert hits.to_numpy().sum() == n

    flag[None] = n
    clear_flag()
    assert impl.get_runtime().prog.get_num_offloaded_tasks_on_last_call() == 2
    assert flag[None] == 0
    assert marks.to_numpy().sum() == n


@test_utils.test(arch=qd.cuda, offline_cache=False)
def test_cuda_device_bound_stored_before_its_loop():
    count = qd.field(qd.i32, shape=())
    out = qd.field(qd.i32, shape=32)

    @qd.kernel
    def store_then_loop(n: qd.i32):
        count[None] = n + 1
        for i in range(count[None]):
            out[i] = n
        count[None] += 1

    for n in (3, 7):
        out.fill(0)
        store_then_loop(n)
        assert count[None] == n + 2
        np.testing.assert_array_equal(out.to_numpy()[: n + 1], n)
        np.testing.assert_array_equal(out.to_numpy()[n + 1 :], 0)


@test_utils.test()
def test_offload_with_cross_block_globals():
    ret = qd.field(qd.f32)

    qd.root.place(ret)

    @qd.kernel
    def ker():
        ret[()] = 0
        for i in range(10):
            ret[()] += i
        ret[()] += 1

    ker()

    assert ret[()] == 46


@test_utils.test(exclude=qd.amdgpu)
def test_offload_with_cross_nested_for():
    @qd.kernel
    def run(a: qd.i32):
        b = a + 1
        for x in range(1):
            for i in range(b):
                print("OK")

    run(2)


@test_utils.test(exclude=qd.amdgpu)
def test_offload_with_cross_if_inside_for():
    @qd.kernel
    def run(a: qd.i32):
        b = a > 2
        for x in range(1):
            if b:
                print("OK")

    run(2)


@test_utils.test(exclude=qd.amdgpu)
def test_offload_with_save():
    a = qd.Vector.field(2, dtype=qd.f32, shape=1)
    b = qd.Vector.field(2, dtype=qd.f32, shape=1)
    c = qd.Vector.field(2, dtype=qd.f32, shape=1)

    @qd.kernel
    def test():
        a[0] = qd.Vector([1, 1])
        b[0] = qd.Vector([0, 0])
        c[0] = qd.Vector([0, 0])
        b[0] += a[0]  # b[0] = [1, 1]
        b[0] /= 2  # b[0] = [0.5, 0.5]
        for i in c:
            c[i] += b[0]  # c[0] = [0.5, 0.5]

    test()
    assert c[0][0] == 0.5
    assert c[0][1] == 0.5
