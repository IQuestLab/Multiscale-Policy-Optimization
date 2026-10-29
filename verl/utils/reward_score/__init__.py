"""BrowseComp-Plus reward entry point."""
from .bc_p_llm_judge import compute_score

async def default_compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if data_source not in {"browsecomp_plus", "browsecomp-plus", "bcp"}:
        raise ValueError(f"Only BrowseComp-Plus is supported: {data_source}")
    return await compute_score(data_source, solution_str, ground_truth, extra_info=extra_info, **kwargs)

_default_compute_score = default_compute_score
