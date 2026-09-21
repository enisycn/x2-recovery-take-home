# Execution root-cause record

Updated 21 September 2026. These findings separate project defects from restrictions of the Codex execution sandbox. No system, driver, Conda or existing robot-workspace setting was changed.

| Symptom | Root cause | Evidence | Resolution in this repository |
| --- | --- | --- | --- |
| Official model `git clone` cannot resolve GitHub | Shell commands run with outbound networking restricted | `Could not resolve host: github.com`; other shell networking is restricted too | Official Git path remains pinned; a safe offline ZIP staging path now validates traversal, symlinks, size, X2 identity, link/joint/collision counts, referenced meshes and hashes |
| Browser cannot download from GitHub or AgiBot docs | A saved browser permission denies both domains | Browser safety review rejected each navigation and prohibited bypassing it | No bypass attempted; the archive must be made available through an allowed browser session or supplied as a local attachment |
| `nvidia-smi` and PyTorch see no GPU | The host has an NVIDIA device and loaded driver, but the sandbox has no `/dev/nvidia*` device nodes | PCI `10de:2c59`, NVIDIA 595.84 modules and CUDA libraries are present; `/dev/nvidia*` is absent; NVML and `torch.cuda` both fail | No driver or host setting is changed. GPU training must run outside this device-isolated sandbox |
| Initial Isaac launcher selected the wrong Python | `isaaclab.sh` falls back to Conda `base` when no environment is active | It selected `<USER_HOME>/miniconda3/bin/python`, which lacked `lazy_loader`/`gymnasium` | Every Isaac entry point now requires explicit `ISAAC_PYTHON`; train/play scope the matching `CONDA_PREFIX` or `VIRTUAL_ENV` to their child process only |
| Concern that no GPU means Isaac physics cannot run | The original smoke script ignored the requested device and used its CUDA default | `SimulationCfg()` defaulted to CUDA even when `--device cpu` was passed | `scripts/probe_isaac_cpu.py` passes the device explicitly; Isaac Sim/PhysX completed five CPU steps successfully |

The remaining dependency is the official X2 archive. Once it is locally available, `stage_agibot_archive.py` and `import_x2_isaac.sh` can import it entirely inside this repository using CPU PhysX. Training can then use CPU with a smaller environment count, or GPU outside the Codex sandbox, without altering the existing robot environment.
