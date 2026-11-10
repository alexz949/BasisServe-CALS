"""Stage labels for diagnostic TP4/TP8 PyTorch profiler traces."""

from functools import wraps

from torch.profiler import record_function

from basisserve.core.llama_tp4_uniform_als import LlamaTP4UniformALSAttention
import benchmarks.system.tp4_joint_sparse as sparse


def labelled(name, function):
    @wraps(function)
    def call(*args, **kwargs):
        with record_function(name):
            return function(*args, **kwargs)
    return call


class ProfileExtension:
    def __init__(self, extension, labels):
        self.extension = extension
        for method, label in labels.items():
            setattr(self, method, labelled(label, getattr(extension, method)))

    def __getattr__(self, name):
        return getattr(self.extension, name)


def install_labels(layers):
    attention = sparse.LlamaTP4JointSparseAttention
    attention._write_codes = labelled("basis.append_codes", attention._write_codes)
    sparse.Metadata.advance = labelled("basis.metadata_update", sparse.Metadata.advance)
    sparse.Metadata.scores = labelled("basis.coarse_scores", sparse.Metadata.scores)
    sparse.candidates = labelled("basis.candidates512", sparse.candidates)
    sparse.slot_indexed_attention = labelled("basis.sparse_attention", sparse.slot_indexed_attention)
    LlamaTP4UniformALSAttention.project_output = labelled(
        "basis.latent_collective_and_decoder", LlamaTP4UniformALSAttention.project_output
    )
    for layer in layers:
        layer.forward = labelled(f"basis.layer_{layer.layer_idx:02d}", layer.forward)
        layer.full_router = ProfileExtension(layer.full_router, {"append": "basis.host_key_append"})
        layer.fine_router = ProfileExtension(
            layer.fine_router, {"conditional_router_page_lse": "basis.fine_router"}
        )
        layer.postprocess_extension = ProfileExtension(
            layer.postprocess_extension, {"select_and_pack": "basis.select_and_pack"}
        )
        layer.slot_extension = ProfileExtension(
            layer.slot_extension, {"refresh": "basis.slot_plan_and_fetch"}
        )
