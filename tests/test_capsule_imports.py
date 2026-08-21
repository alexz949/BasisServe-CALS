"""Cheap CPU-only checks for the extracted public surfaces."""


def test_core_algorithm_imports() -> None:
    from basisserve.core import (  # noqa: F401
        FixedTopKAllGatherOutput,
        LowRankAllReduceOutput,
        PrivateAllGatherOutput,
    )
    import basisserve.core.gqa_routed_ov_joint  # noqa: F401
    import basisserve.core.metric_rank_allocation  # noqa: F401


def test_runtime_imports_do_not_compile_cuda_extension() -> None:
    import basisserve.kernels.ragged_allgather  # noqa: F401
    from basisserve.core.qwen35_gdn_private_ag_runtime import (  # noqa: F401
        Qwen35PrivateAGOutput,
    )
