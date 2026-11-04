
import torch
import numbers
from peft import LoraConfig


# --- LoRA target-module name filters (advisor point 5: parameter efficiency) ---
# Substrings matched against transformer.named_modules() names of the Wan DiT block
# Linears. Actual names verified in wan/modules/model.py WanAttentionBlock:
#   attention projections  : blocks.N.self_attn.{q,k,v,o} (model.py:120-123),
#                            blocks.N.cross_attn.{q,k,v,o} (WanCrossAttention(WanSelfAttention), model.py:159)
#   camera-injection Linears: blocks.N.cam_injector_layer1/cam_injector_layer2 (model.py:220-221),
#                            blocks.N.cam_scale_layer (model.py:222), blocks.N.cam_shift_layer (model.py:223)
#   feed-forward (FFN/MLP)  : blocks.N.ffn.0 / blocks.N.ffn.2 (nn.Sequential of Linears, model.py:213-215)
# (Excluded already by the base predicate: `modulation` nn.Parameter (model.py:218) and any `face` module.)
_ATTN_SUBSTRINGS = ("self_attn", "cross_attn")
_CAM_SUBSTRINGS = ("cam_injector", "cam_scale", "cam_shift")


def get_loraconfig(transformer, rank=128, alpha=128, init_lora_weights="gaussian", targets="all"):
    # Base candidate set: all block Linears except face / modulation (original behavior).
    candidates = []
    for name, module in transformer.named_modules():
        if "blocks" in name and "face" not in name and "modulation" not in name and isinstance(module, torch.nn.Linear):
            candidates.append(name)

    if targets == "all":
        # Original behavior: attention + camera-injection + FFN Linears.
        target_modules = list(candidates)
    elif targets == "attn_cam":
        # Attention projections + camera-injection Linears; EXCLUDE FFN/MLP.
        keep = _ATTN_SUBSTRINGS + _CAM_SUBSTRINGS
        target_modules = [n for n in candidates if any(s in n for s in keep)]
    elif targets == "attn_only":
        # Attention projections only (no FFN, no camera-injection).
        target_modules = [n for n in candidates if any(s in n for s in _ATTN_SUBSTRINGS)]
    else:
        raise ValueError(
            f"Unknown lora targets={targets!r}; expected one of 'all', 'attn_cam', 'attn_only'."
        )

    # Fail loudly if the name filter matched nothing (e.g. module names changed).
    if len(target_modules) == 0:
        example = candidates[:12] if candidates else [
            n for n, m in transformer.named_modules() if isinstance(m, torch.nn.Linear)
        ][:12]
        raise ValueError(
            f"LoRA target filter targets={targets!r} matched 0 modules. "
            f"Substrings attn={_ATTN_SUBSTRINGS} cam={_CAM_SUBSTRINGS} did not match any "
            f"block Linear. Example module names: {example}"
        )

    transformer_lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        init_lora_weights=init_lora_weights,
        target_modules=target_modules,
    )
    return transformer_lora_config



class TensorList(object):

    def __init__(self, tensors):
        """
        tensors: a list of torch.Tensor objects. No need to have uniform shape.
        """
        assert isinstance(tensors, (list, tuple))
        assert all(isinstance(u, torch.Tensor) for u in tensors)
        assert len(set([u.ndim for u in tensors])) == 1
        assert len(set([u.dtype for u in tensors])) == 1
        assert len(set([u.device for u in tensors])) == 1
        self.tensors = tensors
    
    def to(self, *args, **kwargs):
        return TensorList([u.to(*args, **kwargs) for u in self.tensors])
    
    def size(self, dim):
        assert dim == 0, 'only support get the 0th size'
        return len(self.tensors)
    
    def pow(self, *args, **kwargs):
        return TensorList([u.pow(*args, **kwargs) for u in self.tensors])
    
    def squeeze(self, dim):
        assert dim != 0
        if dim > 0:
            dim -= 1
        return TensorList([u.squeeze(dim) for u in self.tensors])
    
    def type(self, *args, **kwargs):
        return TensorList([u.type(*args, **kwargs) for u in self.tensors])
    
    def type_as(self, other):
        assert isinstance(other, (torch.Tensor, TensorList))
        if isinstance(other, torch.Tensor):
            return TensorList([u.type_as(other) for u in self.tensors])
        else:
            return TensorList([u.type(other.dtype) for u in self.tensors])
    
    @property
    def dtype(self):
        return self.tensors[0].dtype
    
    @property
    def device(self):
        return self.tensors[0].device
    
    @property
    def ndim(self):
        return 1 + self.tensors[0].ndim
    
    def __getitem__(self, index):
        return self.tensors[index]
    
    def __len__(self):
        return len(self.tensors)
    
    def __add__(self, other):
        return self._apply(other, lambda u, v: u + v)
    
    def __radd__(self, other):
        return self._apply(other, lambda u, v: v + u)
    
    def __sub__(self, other):
        return self._apply(other, lambda u, v: u - v)
    
    def __rsub__(self, other):
        return self._apply(other, lambda u, v: v - u)
    
    def __mul__(self, other):
        return self._apply(other, lambda u, v: u * v)
    
    def __rmul__(self, other):
        return self._apply(other, lambda u, v: v * u)
    
    def __floordiv__(self, other):
        return self._apply(other, lambda u, v: u // v)
    
    def __truediv__(self, other):
        return self._apply(other, lambda u, v: u / v)
    
    def __rfloordiv__(self, other):
        return self._apply(other, lambda u, v: v // u)
    
    def __rtruediv__(self, other):
        return self._apply(other, lambda u, v: v / u)
    
    def __pow__(self, other):
        return self._apply(other, lambda u, v: u ** v)
    
    def __rpow__(self, other):
        return self._apply(other, lambda u, v: v ** u)
    
    def __neg__(self):
        return TensorList([-u for u in self.tensors])
    
    def __iter__(self):
        for tensor in self.tensors:
            yield tensor
    
    def __repr__(self):
        return 'TensorList: \n' + repr(self.tensors)

    def _apply(self, other, op):
        if isinstance(other, (list, tuple, TensorList)) or (
            isinstance(other, torch.Tensor) and (
                other.numel() > 1 or other.ndim > 1
            )
        ):
            assert len(other) == len(self.tensors)
            return TensorList([op(u, v) for u, v in zip(self.tensors, other)])
        elif isinstance(other, numbers.Number) or (
            isinstance(other, torch.Tensor) and (
                other.numel() == 1 and other.ndim <= 1
            )
        ):
            return TensorList([op(u, other) for u in self.tensors])
        else:
            raise TypeError(
                f'unsupported operand for *: "TensorList" and "{type(other)}"'
            )