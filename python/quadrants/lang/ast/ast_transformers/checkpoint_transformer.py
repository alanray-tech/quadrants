# type: ignore
"""AST recognition and validation for ``qd.checkpoint(...)`` ``with`` blocks.

Lives alongside ``call_transformer.py`` / ``function_def_transformer.py`` so that ``ast_transformer.py`` doesn't have to
grow per-feature. ``ASTTransformer.build_With`` and ``ASTTransformer._is_checkpoint_call`` forward calls into the static
methods here. The implicit no-yield checkpoints of a ``@qd.kernel(graph=True, checkpoints=True)`` kernel are opened by
the ASTBuilder as the body's statements reach the kernel's root block (see
``FunctionDefTransformer.build_FunctionDef``); ``add_implicit_checkpoints`` gives them their entries in the kernel's
per-checkpoint tables.

See ``docs/source/user_guide/graph.md`` for the user-facing surface.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from quadrants.lang.ast.ast_transformer_utils import ASTTransformerFuncContext
from quadrants.lang.exception import QuadrantsSyntaxError


@dataclass
class CheckpointCallInfo:
    """Resolved metadata for a `qd.checkpoint(...)` call recognised in the AST.

    - ``cp_id``: the user-supplied label (an ``int`` or ``IntEnum`` value).
    - ``yield_on_node``: the ``ast.expr`` passed as ``yield_on=`` -- either an ``ast.Name`` (bare kernel parameter, e.g.
      ``flag``) or an ``ast.Attribute`` chain (e.g. ``self.flag`` for a ``@qd.data_oriented`` owner, or ``params.flag``
      where ``params`` is a ``@dataclasses.dataclass`` kernel parameter). ``build_checkpoint_with`` resolves the node
      to a flat C++ arg-id via ``ASTTransformer._resolve_ndarray_kernel_arg_id`` so the runtime can forward it directly
      without per-launch name matching.
    """

    cp_id: int
    yield_on_node: ast.expr


class CheckpointTransformer:
    @staticmethod
    def _looks_like_checkpoint_call(node: ast.expr) -> bool:
        """Cheap structural check: is *node* `qd.checkpoint(...)` or bare `checkpoint(...)`? Doesn't validate args or
        resolve names."""
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        return (isinstance(func, ast.Attribute) and func.attr == "checkpoint") or (
            isinstance(func, ast.Name) and func.id == "checkpoint"
        )

    @staticmethod
    def add_implicit_checkpoints(kernel, num_checkpoints: int) -> None:
        """Give every implicit checkpoint the ASTBuilder opened (ids up to *num_checkpoints* - 1 not yet in the kernel's
        per-checkpoint tables) its entry: no ``yield_on`` and no user label, so it never yields and is never a resume
        target."""
        while len(kernel.checkpoint_yield_on_args) < num_checkpoints:
            kernel.checkpoint_yield_on_args.append(None)
            kernel.checkpoint_yield_on_cpp_arg_ids.append(-1)
            kernel.checkpoint_user_labels_by_cp_id.append(None)

    @staticmethod
    def _resolve_cp_id(node: ast.expr, global_vars: dict) -> int:
        """Resolve the first positional arg of `qd.checkpoint(cp_id, ...)` to a Python int (or IntEnum instance).

        Accepts (a) `ast.Constant` int literals (``qd.checkpoint(0, ...)``), (b) `ast.Attribute` references to `IntEnum`
        values resolved against the kernel's `global_vars` (``qd.checkpoint(Stage.SIM, ...)`` where ``Stage`` is an
        `IntEnum` defined at module scope), and (c) `ast.Name` references to module-level int constants
        (``qd.checkpoint(CP_LOAD, ...)`` where ``CP_LOAD`` is an int defined at module scope). For (b) and (c) we return
        the resolved value AS-IS (without re-wrapping through `int(...)`) so an IntEnum member identity is preserved
        end-to-end -- the user writes `qd.checkpoint(Stage.SIM, ...)`, then reads `status.checkpoint` and gets back
        `Stage.SIM`, not the raw int. Rejects everything else with a clear error so the user gets a compile-time
        diagnostic rather than a confusing template-mapper failure later.
        """
        # Plain `qd.checkpoint(0, ...)` literal.
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            return node.value
        # `qd.checkpoint(Stage.SIM, ...)` -- look up `Stage` in the kernel's module globals, then `.SIM` on it.
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            container_name = node.value.id
            if container_name in global_vars:
                container = global_vars[container_name]
                if hasattr(container, node.attr):
                    val = getattr(container, node.attr)
                    if isinstance(val, int) and not isinstance(val, bool):
                        return val
        # `qd.checkpoint(CP_LOAD, ...)` -- bare name referencing a module-level int constant.
        if isinstance(node, ast.Name) and node.id in global_vars:
            val = global_vars[node.id]
            if isinstance(val, int) and not isinstance(val, bool):
                return val
        raise QuadrantsSyntaxError(
            "qd.checkpoint() first argument must be an int literal or an IntEnum value (e.g. `Stage.SIM`). Got an "
            f"unresolvable expression at line {getattr(node, 'lineno', '?')}; the cp_id must be statically "
            "determinable at AST-walk time so the framework can build the label -> internal cp_id map and the "
            "host-side resume API can refer to checkpoints by name."
        )

    @staticmethod
    def is_checkpoint_call(node: ast.expr, global_vars: dict) -> CheckpointCallInfo | None:
        """If *node* is a `qd.checkpoint(...)` call return a `CheckpointCallInfo`; otherwise return `None`.

        Validates the call shape and raises `QuadrantsSyntaxError` for misuse so the user gets a clear message at the
        `with` site rather than a vague "not stream_parallel" error later.
        """
        if not CheckpointTransformer._looks_like_checkpoint_call(node):
            return None
        # User-written `qd.checkpoint(cp_id, yield_on)` -- both args are required.
        if len(node.args) + len(node.keywords) == 0:
            raise QuadrantsSyntaxError("qd.checkpoint() takes two arguments: `qd.checkpoint(cp_id, yield_on=flag)`.")
        # Collect cp_id (positional 0 or kw `cp_id=`) and yield_on (positional 1 or kw `yield_on=`).
        cp_id_arg: ast.expr | None = None
        yield_on_arg: ast.expr | None = None
        for i, arg in enumerate(node.args):
            if i == 0:
                cp_id_arg = arg
            elif i == 1:
                yield_on_arg = arg
            else:
                raise QuadrantsSyntaxError(
                    f"qd.checkpoint() takes at most 2 positional arguments (cp_id, yield_on); got {len(node.args)}"
                )
        for kw in node.keywords:
            if kw.arg == "cp_id":
                if cp_id_arg is not None:
                    raise QuadrantsSyntaxError("qd.checkpoint() got `cp_id` both positionally and as a keyword")
                cp_id_arg = kw.value
            elif kw.arg == "yield_on":
                if yield_on_arg is not None:
                    raise QuadrantsSyntaxError("qd.checkpoint() got `yield_on` both positionally and as a keyword")
                yield_on_arg = kw.value
            else:
                raise QuadrantsSyntaxError(
                    f"qd.checkpoint() got unexpected keyword argument {kw.arg!r}; only 'cp_id' and 'yield_on' are "
                    "supported"
                )
        if cp_id_arg is None:
            raise QuadrantsSyntaxError(
                "qd.checkpoint() is missing required argument `cp_id` (e.g. `qd.checkpoint(0, yield_on=flag)` or "
                "`qd.checkpoint(Stage.SIM, yield_on=flag)`)"
            )
        if yield_on_arg is None:
            raise QuadrantsSyntaxError(
                "qd.checkpoint() is missing required argument `yield_on` (e.g. "
                "`qd.checkpoint(0, yield_on=overflow_flag)`)"
            )
        # `yield_on=` must point at an ndarray kernel argument -- a bare parameter (`yield_on=flag`), a
        # `@qd.data_oriented` member (`yield_on=self.flag`), or a `@dataclasses.dataclass` parameter member
        # (`yield_on=params.flag`). Other expressions can't be lowered to a flat arg-id and are rejected here so the
        # user gets a clear compile-time error at the `with` site.
        if not isinstance(yield_on_arg, (ast.Name, ast.Attribute)):
            raise QuadrantsSyntaxError(
                "qd.checkpoint(yield_on=...) must reference a kernel ndarray argument -- e.g. `yield_on=flag` for "
                "a bare parameter, `yield_on=self.flag` for a @qd.data_oriented member, or `yield_on=params.flag` "
                "for a @dataclasses.dataclass parameter member; arbitrary expressions are not supported"
            )
        cp_id_value = CheckpointTransformer._resolve_cp_id(cp_id_arg, global_vars)
        return CheckpointCallInfo(cp_id=cp_id_value, yield_on_node=yield_on_arg)

    @staticmethod
    def build_checkpoint_with(
        ctx: ASTTransformerFuncContext,
        node: ast.With,
        info: CheckpointCallInfo,
        build_stmts,
    ) -> None:
        """Handles a `with qd.checkpoint(...):` block.

        Validates the use-site (kernel must be `graph=True, checkpoints=True`, no nesting, the `yield_on` arg must be a
        kernel parameter and the `cp_id` must be unique across the kernel) and appends an entry to
        `kernel.checkpoint_yield_on_args` + `kernel.checkpoint_user_labels_by_cp_id`. Walks the body transparently --
        every statement inside the `with`, inlined func bodies and bare statements included, carries the checkpoint's
        internal `cp_id` into the tasks it lowers to (a side-effect-free bare statement stays always-run, see
        `assemble_serial_statements` in `quadrants/transforms/offload.cpp`). The internal `cp_id` is assigned by
        declaration order (the C++ `ASTBuilder.begin_checkpoint()` counter, mirrored as the list index in
        `checkpoint_yield_on_args`); implicit checkpoints take the ids in between.
        """
        if not ctx.is_kernel:
            raise QuadrantsSyntaxError("qd.checkpoint() can only be used inside @qd.kernel, not @qd.func")
        kernel = ctx.global_context.current_kernel
        if not kernel.use_graph:
            raise QuadrantsSyntaxError(
                "qd.checkpoint() requires @qd.kernel(graph=True); the resume model only applies to graph kernels"
            )
        if not kernel.use_checkpoints:
            raise QuadrantsSyntaxError(
                f"qd.checkpoint() found in kernel {kernel.func.__name__!r}, but this kernel was not decorated with "
                "`checkpoints=True`. Add `checkpoints=True` to the decorator -- e.g. "
                "`@qd.kernel(graph=True, checkpoints=True)` -- to opt into the pause / resume model."
            )
        if getattr(ctx, "_in_checkpoint", False):
            raise QuadrantsSyntaxError(
                "qd.checkpoint() cannot be nested inside another qd.checkpoint(); checkpoints in the "
                "same kernel must be flat siblings (a checkpoint inside qd.graph.do_while is fine)"
            )

        # Resolve `yield_on=` (a bare parameter or `@qd.data_oriented` / `@dataclasses.dataclass` member ndarray) to its
        # flat C++ arg-id at AST-build time. ``resolve_ndarray_kernel_arg_id`` raises a user-facing
        # ``QuadrantsSyntaxError`` if the expression does not name a real ndarray kernel argument, which keeps the
        # diagnostic at the `with` site instead of leaking into the launcher. Both the label (for
        # ``checkpoint_yield_on_args`` / introspection) and the resolved arg-id (for the runtime) are stashed and
        # forwarded to the launch path below.
        # pylint: disable-next=C0415,import-outside-toplevel
        from quadrants.lang.ast.ast_transformers.ndarray_arg_resolver import (
            resolve_ndarray_kernel_arg_id,
        )

        yield_on_label, yield_on_cpp_arg_id = resolve_ndarray_kernel_arg_id(
            ctx, kernel, info.yield_on_node, "qd.checkpoint(yield_on=...)"
        )
        # Reject duplicate user-supplied cp_id labels.
        existing = [lbl for lbl in kernel.checkpoint_user_labels_by_cp_id if lbl is not None]
        if info.cp_id in existing:
            raise QuadrantsSyntaxError(
                f"qd.checkpoint(cp_id={info.cp_id!r}, ...) declares a cp_id that is already used by another "
                f"checkpoint in kernel {kernel.func.__name__!r}. Each `cp_id` label must be unique so the host "
                "loop can map `status.checkpoint` and `kernel.resume(from_checkpoint=...)` unambiguously."
            )

        # Hand control to the C++ ASTBuilder so that every statement and for-loop emitted by `build_stmts` below is
        # tagged with this checkpoint's internal `cp_id`. The C++ counter is the source of truth for the dense internal
        # cp_id (it closes the implicit checkpoint open before this one). The implicit checkpoints it handed out since
        # the last explicit one get their entries first, and the Python list index is cross-checked against it so that
        # a future refactor that misaligns the two surfaces fires immediately.
        cpp_cp_id = ctx.ast_builder.begin_checkpoint()
        CheckpointTransformer.add_implicit_checkpoints(kernel, cpp_cp_id)
        kernel.checkpoint_yield_on_args.append(yield_on_label)
        kernel.checkpoint_yield_on_cpp_arg_ids.append(yield_on_cpp_arg_id)
        kernel.checkpoint_user_labels_by_cp_id.append(info.cp_id)
        py_cp_id = len(kernel.checkpoint_yield_on_args) - 1
        assert cpp_cp_id == py_cp_id, (
            f"C++ ASTBuilder.begin_checkpoint() returned cp_id={cpp_cp_id} but Python "
            f"kernel.checkpoint_yield_on_args index expected {py_cp_id}; these counters "
            f"must stay in lockstep so the GraphManager can index yield_on by cp_id"
        )
        ctx._in_checkpoint = True
        try:
            build_stmts(ctx, node.body)
        finally:
            ctx._in_checkpoint = False
            ctx.ast_builder.end_checkpoint()
        return None
