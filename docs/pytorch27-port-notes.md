# PyTorch 1.10 to 2.7 Porting Notes

## Summary

The codebase required minimal changes for PyTorch 2.7 compatibility. The primary
breaking change is that `torch.load()` now requires an explicit `weights_only`
parameter (defaulting to `True` since PyTorch 2.6). All other APIs used in this
codebase remain compatible.

## Changes by File

### `src/turtlebot3_drl/turtlebot3_drl/common/storagemanager.py`

**1. `network_load_weights()` (line 76) -- `torch.load()` requires `weights_only` parameter**

- Old: `torch.load(filepath, self.map_location)`
- New: `torch.load(filepath, self.map_location, weights_only=True)`
- Rationale: Network state dicts contain only tensors and primitive types, so
  `weights_only=True` is safe and preferred for security (prevents arbitrary
  code execution during unpickling).

**2. `CpuUnpickler.find_class()` (line 109) -- `torch.load()` requires `weights_only` parameter**

- Old: `torch.load(io.BytesIO(b), map_location=self.map_location)`
- New: `torch.load(io.BytesIO(b), map_location=self.map_location, weights_only=True)`
- Rationale: This lambda is used inside `CpuUnpickler` to remap torch storage
  objects when loading pickled models across devices. The torch storage bytes
  only contain tensor data, so `weights_only=True` is safe.

### Files with No Changes Required

- **`off_policy_agent.py`** -- All APIs (`torch.from_numpy`, `torch.optim.AdamW`,
  `torch.nn.functional.smooth_l1_loss`, `torch.nn.init.xavier_uniform_`,
  `super(Network, self).__init__()`) remain valid in PyTorch 2.7.

- **`ddpg.py`** -- All APIs (`nn.Linear`, `torch.relu`, `torch.tanh`,
  `nn.utils.clip_grad_norm_`, `torch.clamp`, `torch.add`) remain valid.

- **`td3.py`** -- All APIs (`torch.randn_like`, `nn.utils.clip_grad_norm_`,
  `torch.min`) remain valid.

- **`dqn.py`** -- All APIs (`F.mse_loss`, `torch.unsqueeze`, `.amax()`,
  `.gather()`, `nn.utils.clip_grad_norm_`) remain valid.

- **`drl_agent.py`** -- No direct PyTorch API calls (delegates to model and
  storage manager).

- **`replaybuffer.py`** -- Pure Python/NumPy, no PyTorch usage.

- **`utilities.py`** -- `torch.cuda.is_available()`, `torch.cuda.get_device_name()`,
  and `torch.device()` remain valid in PyTorch 2.7.

## Notes on `torch.load()` `weights_only` Parameter

Starting in PyTorch 2.0, `torch.load()` began emitting a `FutureWarning` when
`weights_only` was not specified. As of PyTorch 2.6, the default changed to
`weights_only=True`, causing loads of arbitrary pickled objects to fail unless
explicitly opted out with `weights_only=False`.

For this codebase:
- **Model state dicts** (`.pt` files saved via `torch.save(network.state_dict(), path)`)
  contain only tensors and safe primitive types. Use `weights_only=True`.
- **Pickled model objects** (`.pkl` files) are loaded via Python's `pickle` module
  directly, not via `torch.load()`, so they are unaffected by this change.
- The **`CpuUnpickler`** class intercepts `torch.storage._load_from_bytes` calls
  during unpickling to remap device locations. The inner `torch.load()` call
  there only processes raw tensor storage bytes, so `weights_only=True` is safe.

## APIs Verified Compatible (PyTorch 1.10 to 2.7)

| API | Status |
|-----|--------|
| `torch.nn.Linear` | No change |
| `torch.nn.Module` | No change |
| `torch.nn.functional.smooth_l1_loss` | No change |
| `torch.nn.functional.mse_loss` | No change |
| `torch.nn.utils.clip_grad_norm_` | No change |
| `torch.nn.init.xavier_uniform_` | No change |
| `torch.optim.AdamW` | No change |
| `torch.relu` / `torch.tanh` | No change |
| `torch.from_numpy` | No change |
| `torch.randn_like` | No change |
| `torch.clamp` / `torch.add` / `torch.min` | No change |
| `torch.save` | No change |
| `torch.cuda.is_available` | No change |
| `torch.cuda.get_device_name` | No change |
| `torch.device` | No change |
| `Tensor.detach().cpu()` | No change |
| `Tensor.data.copy_()` | No change |
| `Tensor.amax()` / `Tensor.gather()` | No change |
