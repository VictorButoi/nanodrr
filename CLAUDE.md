# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install dependencies:**
```bash
uv sync --group dev        # Development dependencies (pre-commit)
uv sync --group docs       # Documentation dependencies (mkdocs, etc.)
uv sync --extra scene      # Optional 3D visualization (pyvista, vtk)
```

**Lint and format:**
```bash
uv run ruff check --fix    # Run linter with auto-fix
uv run ruff format         # Format code
```

**Run benchmarks:**
```bash
cd tests/benchmark && bash main.sh   # Full benchmark suite
uv run python tests/benchmark/benchmark.py  # Standalone benchmark
```

**Docs (local dev):**
```bash
uv run mkdocs serve        # Serve docs locally
```

**Pre-commit hooks:**
```bash
uv run pre-commit run --all-files
```

There are no traditional pytest unit tests. Testing is done via performance benchmarks in `tests/benchmark/` and by executing `docs/tutorials/demo.ipynb` in CI.

## Architecture

**nanodrr** renders synthetic X-ray images (DRRs) from 3D CT volumes via differentiable ray marching. The library is ~5-8x faster than DiffDRR and is fully compatible with `torch.compile` and mixed precision.

### Module Overview (`src/nanodrr/`)

- **`data/`** — Core data container (`Subject`) that wraps CT volume + labelmap tensors, pre-computes coordinate frame transforms, and converts Hounsfield Units to attenuation coefficients
- **`camera/`** — Pinhole camera construction following Hartley & Zisserman:
  - `intrinsics.py`: `make_k_inv()` builds the inverse intrinsic matrix from physical C-arm parameters (SDD, pixel spacing, principal point)
  - `extrinsics.py`: `make_rt_inv()` builds camera-to-world transforms from Euler angles + translation, rotating around the subject isocenter
- **`drr/`** — Core rendering engine:
  - `render.py`: Functional `render()` — differentiable ray-marching via `grid_sample`
  - `drr.py`: `DRR` class (torch.nn.Module) wrapping `render()` with fixed intrinsics
- **`geometry/`** — SE(3) transformation utilities and parameterization conversion (`se3.py`): Euler angles, quaternions, rotation matrices, SO(3)/SE(3) log-space
- **`metrics/`** — Image similarity losses: NCC 2D, gradient NCC, multiscale NCC, geodesic SE(3) distance
- **`registration/`** — Base `Registration` class for 2D/3D pose optimization with learnable SO(3) log parameterization
- **`scene/`** — 3D visualization via PyVista (optional `[scene]` dependency)
- **`plot/`** — Matplotlib DRR visualization utilities

### Rendering Pipeline

```
Pixel coords → (k_inv) → Camera rays → (rt_inv) → World space
    → (Subject.world_to_grid) → Normalized grid [-1,1]³
    → grid_sample along rays → integrate intensities → (B, C, H, W)
```

The `Subject` class pre-fuses all spatial transforms into a single `world_to_grid` matrix to minimize matrix multiplications at render time.

### Key Design Patterns

- **Functional + OOP**: Core `render()` is functional; `DRR` class wraps it for fixed-intrinsic workflows
- **jaxtyping**: All tensor shapes are annotated in function signatures (e.g., `Float[Tensor, "B C H W"]`)
- **Modular**: Subject, intrinsics, and extrinsics are fully independent and can be mixed at runtime
- **Batching**: All ops batch across dim 0; intrinsics can vary per image in a batch (e.g., biplane C-arms)

### Two Primary Usage Interfaces

```python
# Functional (flexible intrinsics per call)
from nanodrr.drr import render
img = render(subject, k_inv, rt_inv, sdd, height, width, n_samples=500)

# Class-based (fixed intrinsics)
from nanodrr.drr import DRR
drr = DRR.from_carm_intrinsics(sdd, delx, dely, x0, y0, height, width)
img = drr(subject, rt_inv)
```

### Ruff Configuration Note

`F722` and `F821` are suppressed project-wide because jaxtyping uses forward annotations that trigger false positives.
