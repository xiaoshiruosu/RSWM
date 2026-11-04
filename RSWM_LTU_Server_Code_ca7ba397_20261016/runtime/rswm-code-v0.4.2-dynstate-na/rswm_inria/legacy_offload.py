"""CPU storage for frozen Wan weights, including a real autograd path.

Canonical Parameters keep their identity. Trainable LoRA Parameters stay on the
execution device. While a module executes, its frozen parameter slots expose
independent detached staging Tensors; the original CPU Parameters are restored
afterward. This matters because non-reentrant checkpointing may cache a frozen
Tensor object directly: changing that object's .data device breaks metadata.
Saved-tensor hooks retain CPU tensors and restore them only when autograd needs
them. This also supports the original non-reentrant block checkpointing: its
inner hooks may retain one recomputed block, not all expert weights.

This module does not claim that a particular video fits a 40 GiB GPU. CPU RAM,
PCIe traffic, activation size and the actual attention kernel must be measured
on the allocated server. Transfers are deliberately synchronous: a storage is
never evicted while a transfer or a weight mutation check is still pending.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import threading
import weakref


@dataclass
class _Frozen:
    name: str
    parameter: object
    cpu: object
    count: int = 0
    staged: object = None
    staged_storage: object = None
    staged_layout: object = None
    bindings: list = field(default_factory=list)


@dataclass
class _Saved:
    cpu: object
    device: object


class FrozenCPUOffload:
    """Install frozen-weight staging without changing LoRA/optimizer identity.

    ``forward_context`` must surround every expert forward. The hooks remain
    installed through backward, so checkpoint recomputation stages parameters
    as well. Context exit does not invalidate saved tensors. Quiesce before a
    checkpoint/state load or failure recovery; rebind after replacing modules.
    A CPU execution device is supported for mathematical regression tests only.
    """

    def __init__(self, model, *, device=None, target_device=None, dtype=None):
        import torch

        if device is not None and target_device is not None:
            raise ValueError("Supply one offload execution device")
        selected = target_device if target_device is not None else device
        if selected is None:
            raise ValueError("Explicit offload execution device required")
        self.device = torch.device(selected)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("Frozen CPU offload supports CPU simulation or CUDA")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("Allocated CUDA execution device unavailable")
        self.model = model
        self.dtype = dtype
        self._handles = []
        self._entries = {}
        self._owners = {}
        self._leases = {}
        self._staged_storages = {}
        self._saved_cache = {}
        self._buffers = {}
        self._closed = False
        self._poison = None
        self._thread = None
        self._contexts = 0
        self._stats = {
            "weight_onload_bytes": 0, "weight_eviction_check_bytes": 0,
            "saved_tensor_cpu_bytes": 0, "saved_tensor_restore_bytes": 0,
            "saved_tensor_reused_bytes": 0,
            "frozen_saved_view_bytes": 0, "module_onloads": 0,
            "maximum_staged_frozen_bytes": 0,
        }
        if getattr(model, "_rswm_frozen_cpu_offload", None) is not None:
            raise ValueError("Expert already has an offload controller; close before rebinding")

        parameters = list(model.named_parameters())
        if not parameters:
            raise ValueError("Expert has no parameters")
        # Validate before changing any storage. No full-model .to(cuda) occurs.
        for name, parameter in parameters:
            if parameter.requires_grad and "lora_" not in name.lower():
                raise ValueError("CPU offload permits trainable LoRA parameters only: " + name)
            if not parameter.requires_grad and parameter.grad is not None:
                raise ValueError("Frozen parameter carries a gradient: " + name)
            if parameter.layout != torch.strided:
                raise ValueError("CPU offload requires dense strided parameters")

        with torch.no_grad():
            for name, parameter in parameters:
                target_dtype = dtype if dtype is not None and parameter.is_floating_point() else parameter.dtype
                if parameter.requires_grad:
                    parameter.data = parameter.detach().to(device=self.device, dtype=target_dtype)
                    if parameter.grad is not None:
                        parameter.grad.data = parameter.grad.to(device=self.device, dtype=target_dtype)
                else:
                    master = parameter.detach().to(device="cpu", dtype=target_dtype, copy=True)
                    parameter.data = master
                    self._entries[id(parameter)] = _Frozen(name, parameter, master)

        try:
            self._handles.append(model.register_forward_pre_hook(self._root_guard, prepend=True))
            for module in model.modules():
                entries = []
                seen = set()
                # Include aliased parameter names, even though
                # named_parameters normally deduplicates their objects.
                for name, parameter in module._parameters.items():
                    entry = self._entries.get(id(parameter))
                    if entry is not None:
                        entry.bindings.append((module, name))
                        if id(parameter) not in seen:
                            entries.append(entry)
                            seen.add(id(parameter))
                buffers = []
                for name, buffer in module.named_buffers(recurse=False):
                    if buffer is not None:
                        cpu = buffer.detach().to("cpu", copy=True)
                        module._buffers[name] = cpu
                        self._buffers[(id(module), name)] = cpu
                        buffers.append(name)
                if entries or buffers:
                    self._owners[id(module)] = (module, entries, buffers)
                    self._leases[id(module)] = []
                    self._handles.append(module.register_forward_pre_hook(self._before))
                    self._handles.append(module.register_forward_hook(self._after, always_call=True))

            base = model.get_base_model() if hasattr(model, "get_base_model") else model
            patch = getattr(base, "patch_embedding", None)
            self._primed = self._entries.get(id(getattr(patch, "weight", None)))
            self._prime_leases = []
            if self._primed is not None:
                # Original Wan reads this weight's device before calling its
                # embedding submodule and before moving the RoPE frequencies.
                self._handles.append(base.register_forward_pre_hook(self._prime, prepend=True))
                self._handles.append(base.register_forward_hook(self._unprime, always_call=True))
            model._rswm_frozen_cpu_offload = self
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _bytes(tensor):
        return tensor.numel() * tensor.element_size()

    @staticmethod
    def _storage(tensor):
        return (tensor.device, tensor.dtype, tensor.untyped_storage().data_ptr(),
                tensor.untyped_storage().nbytes())

    def _check(self):
        if self._closed:
            raise RuntimeError("Frozen CPU offload controller is closed")
        if self._poison is not None:
            raise RuntimeError("Frozen CPU offload failed: " + self._poison)

    def _root_guard(self, module, inputs):
        self._check()
        if self._contexts == 0:
            raise RuntimeError("Expert forward requires offload forward_context")

    def _acquire(self, entry):
        import torch
        if entry.parameter.requires_grad:
            raise RuntimeError("Frozen parameter became trainable: " + entry.name)
        if entry.count == 0:
            # A new CPU allocation in state restoration invalidates the old
            # master. The caller must close/rebind rather than use stale data.
            if self._storage(entry.parameter) != self._storage(entry.cpu):
                raise RuntimeError("Frozen storage changed; rebind offload after restore: " + entry.name)
            if any(module._parameters[name] is not entry.parameter for module, name in entry.bindings):
                raise RuntimeError("Frozen parameter binding changed; rebind offload: " + entry.name)
            with torch.no_grad():
                entry.staged = entry.cpu.to(self.device, copy=True)
                # Do not mutate the canonical Parameter's .data. PyTorch 2.10
                # checkpoint's recomputation pack hook detaches only tensors
                # with requires_grad=True. A frozen Parameter would therefore
                # be cached by object identity, and a later .data->CPU would
                # also change the cached Tensor's device. Independent detached
                # staging Tensors remain on their original device after the
                # module slot is restored, as functional parameter substitution
                # does. Optimizer-owned LoRA Parameters are never substituted.
                for module, name in entry.bindings:
                    module._parameters[name] = entry.staged
            entry.staged_storage = self._storage(entry.staged)
            entry.staged_layout = (tuple(entry.staged.shape), tuple(entry.staged.stride()), entry.staged.storage_offset())
            self._staged_storages[entry.staged_storage] = entry
            self._stats["weight_onload_bytes"] += self._bytes(entry.cpu)
            resident = sum(self._bytes(e.cpu) for e in self._entries.values() if e.staged is not None)
            self._stats["maximum_staged_frozen_bytes"] = max(
                self._stats["maximum_staged_frozen_bytes"], resident)
        entry.count += 1

    def _release(self, entry, *, force=False):
        import torch
        if entry.count == 0:
            return
        entry.count = 0 if force else entry.count - 1
        if entry.count:
            return
        staged = entry.staged
        error = None
        try:
            # Never hide a .data edit by restoring the untouched master.
            # Exact bytes catch even signed-zero edits and retain NaN payloads.
            binding_changed = any(module._parameters[name] is not staged for module, name in entry.bindings)
            storage_changed = self._storage(staged) != entry.staged_storage
            layout_changed = (tuple(staged.shape), tuple(staged.stride()), staged.storage_offset()) != entry.staged_layout
            current = staged.detach().to("cpu", copy=True).contiguous()
            self._stats["weight_eviction_check_bytes"] += self._bytes(current)
            expected = entry.cpu.contiguous()
            if binding_changed or storage_changed or layout_changed or current.shape != expected.shape or current.dtype != expected.dtype or not torch.equal(
                    current.reshape(-1).view(torch.uint8), expected.reshape(-1).view(torch.uint8)):
                error = "Frozen parameter changed during forward: " + entry.name
                self._poison = error
        finally:
            self._staged_storages.pop(entry.staged_storage, None)
            with torch.no_grad():
                for module, name in entry.bindings:
                    module._parameters[name] = entry.parameter
            entry.staged = None
            entry.staged_storage = None
            entry.staged_layout = None
        if error:
            raise RuntimeError(error)

    def _before(self, module, inputs):
        self._check()
        _, entries, buffers = self._owners[id(module)]
        acquired = []
        self._leases[id(module)].append(acquired)
        try:
            for entry in entries:
                self._acquire(entry)
                acquired.append(entry)
            for name in buffers:
                module._buffers[name] = self._buffers[(id(module), name)].to(self.device, copy=True)
            self._stats["module_onloads"] += 1
        except BaseException:
            self._evict_owner(module)
            raise

    def _evict_owner(self, module):
        _, _, buffers = self._owners[id(module)]
        stack = self._leases[id(module)]
        if not stack:
            return
        entries = stack.pop()
        first_error = None
        for entry in reversed(entries):
            try:
                self._release(entry)
            except BaseException as exc:
                first_error = first_error or exc
        # Buffers may legitimately update (e.g. a counter). Preserve them;
        # frozen backbone mutation protection applies to Parameters above.
        for name in buffers:
            current = module._buffers[name]
            cpu = current.detach().to("cpu", copy=True)
            self._buffers[(id(module), name)] = cpu
            module._buffers[name] = cpu
        if first_error is not None:
            raise first_error

    def _after(self, module, inputs, output):
        self._evict_owner(module)

    def _prime(self, module, inputs):
        self._check()
        self._prime_leases.append(False)
        self._acquire(self._primed)
        self._prime_leases[-1] = True

    def _unprime(self, module, inputs, output):
        if self._prime_leases and self._prime_leases.pop():
            self._release(self._primed)

    def _pack(self, tensor):
        self._check()
        entry = self._staged_storages.get(self._storage(tensor))
        if entry is not None:
            offset = tensor.storage_offset() - entry.staged.storage_offset() + entry.cpu.storage_offset()
            cpu = entry.cpu.as_strided(tensor.size(), tensor.stride(), offset).detach()
            self._stats["frozen_saved_view_bytes"] += self._bytes(tensor)
        else:
            # copy=True is necessary for CPU simulations as well: saving an
            # alias of a trainable parameter would let optimizer writes change
            # the values needed by a pending backward pass.
            # The original checkpoint passes the same large time/camera
            # tensors to every block. Reuse their immutable CPU snapshot, but
            # only while this exact source Tensor is alive and unchanged.
            # Storage pointer alone is unsafe: CUDA's allocator reuses it.
            key = (id(tensor), self._storage(tensor), tensor._version,
                   tuple(tensor.shape), tuple(tensor.stride()), tensor.storage_offset())
            cached = self._saved_cache.get(key)
            cpu = cached[1]() if cached is not None and cached[0]() is tensor else None
            if cpu is not None:
                self._stats["saved_tensor_reused_bytes"] += self._bytes(cpu)
            else:
                cpu = tensor.detach().to("cpu", copy=True)
                self._stats["saved_tensor_cpu_bytes"] += self._bytes(cpu)
                owner = weakref.ref(self)

                def forget(reference, key=key, owner=owner):
                    controller = owner()
                    if controller is not None:
                        current = controller._saved_cache.get(key)
                        if current is not None and any(item is reference for item in current):
                            controller._saved_cache.pop(key, None)

                self._saved_cache[key] = (weakref.ref(tensor, forget), weakref.ref(cpu, forget))
        return _Saved(cpu, tensor.device)

    def _unpack(self, saved):
        self._check()
        self._stats["saved_tensor_restore_bytes"] += self._bytes(saved.cpu)
        # Do not cache device copies in the saved object: doing so would retain
        # every frozen weight after the first backward traversal.
        return saved.cpu.to(saved.device, copy=True)

    @contextmanager
    def forward_context(self):
        import torch
        self._check()
        thread = threading.get_ident()
        if self._thread is not None and self._thread != thread:
            raise RuntimeError("Concurrent expert offload forwards are unsupported")
        self._thread = thread
        self._contexts += 1
        try:
            with torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack):
                yield self
        finally:
            self._contexts -= 1
            if self._contexts == 0:
                try:
                    self.quiesce()
                finally:
                    self._thread = None

    def quiesce(self):
        """Return every frozen storage to CPU, including exceptional forwards."""
        first_error = None
        for module, _, _ in self._owners.values():
            while self._leases[id(module)]:
                try:
                    self._evict_owner(module)
                except BaseException as exc:
                    first_error = first_error or exc
        self._prime_leases.clear()
        for entry in self._entries.values():
            try:
                self._release(entry, force=True)
            except BaseException as exc:
                first_error = first_error or exc
        if first_error is not None:
            raise first_error

    def report(self):
        return {
            "strategy": "frozen_parameter_cpu_saved_tensors",
            "execution_device": str(self.device),
            "cpu_simulation": self.device.type == "cpu",
            "real_40gb_model_validation": False,
            "frozen_parameter_bytes": sum(self._bytes(e.cpu) for e in self._entries.values()),
            "trainable_parameter_bytes": sum(self._bytes(p) for p in self.model.parameters() if p.requires_grad),
            "staged_frozen_bytes": sum(self._bytes(e.cpu) for e in self._entries.values() if e.staged is not None),
            "closed": self._closed,
            "failure": self._poison,
            "saved_cpu_snapshot_cache_entries": len(self._saved_cache),
            **self._stats,
        }

    def close(self):
        if self._closed:
            return
        try:
            self.quiesce()
        finally:
            for handle in self._handles:
                handle.remove()
            self._handles.clear()
            self._saved_cache.clear()
            if getattr(self.model, "_rswm_frozen_cpu_offload", None) is self:
                del self.model._rswm_frozen_cpu_offload
            self._closed = True


def attach_frozen_cpu_offload(model, *, device=None, target_device=None, dtype=None):
    """Close stale hooks before rebinding an explicitly restored expert."""
    previous = getattr(model, "_rswm_frozen_cpu_offload", None)
    if previous is not None:
        previous.close()
    return FrozenCPUOffload(model, device=device, target_device=target_device, dtype=dtype)


def rswm_expert_forward(model, *args, **kwargs):
    """Prepared original inference entry point, with no whole-expert .to()."""
    controller = getattr(model, "_rswm_frozen_cpu_offload", None)
    if controller is None:
        raise RuntimeError("Explicit block CPU offload expert controller missing")
    with controller.forward_context():
        return model(*args, **kwargs)
