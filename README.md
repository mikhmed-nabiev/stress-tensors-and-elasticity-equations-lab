# Project 7: stress tensor and elasticity equations (Lab 1)

This project fits a hyperelastic strain energy and its variants to three
published data sources. A hyperelastic material stores all deformation work
as elastic energy. The strain energy function W(F) defines its stress.

The notebook `notebooks/lab1_hyperelasticity.ipynb` presents the
computational experiment. The package `hyperfit` in `src/` holds all logic.
The folder `tests/` holds the unit tests.

## Contents

| Path | Content |
|---|---|
| `notebooks/lab1_hyperelasticity.ipynb` | The experiment: equations, graphs, fits, errors and plots |
| `src/hyperfit/sources.py` | HTTP URLs, pinned commits, SHA-256 checksums and citations |
| `src/hyperfit/data.py` | Download, cache, parsers and preprocessing |
| `src/hyperfit/kinematics.py` | Deformation gradient, invariants and stretches (torch) |
| `src/hyperfit/trees.py` | DEAP expression trees, SymPy conversion and zss tree edit distance |
| `src/hyperfit/models.py` | The `ModelVariant` class and the strain energy variants |
| `src/hyperfit/mechanics.py` | Stress and rank-4 tangent by automatic differentiation |
| `src/hyperfit/fem.py` | Finite element energy, weak-form residual and forward solver |
| `src/hyperfit/parameters.py` | Parameter vectors, bounds and expected values |
| `src/hyperfit/fitting.py` | Linear least squares, Levenberg-Marquardt and CMA-ES |
| `src/hyperfit/graph.py` | The governing equation as a directed acyclic graph (networkx) |
| `src/hyperfit/evaluation.py` | Error metrics and physical checks |
| `src/hyperfit/experiment.py` | Steps of the experiment, used by the notebook |
| `src/hyperfit/plotting.py` | Figures |
| `outputs/figures/` | The figures of the last notebook run |

## Setup

The project needs Python 3.12 or newer.

### Option 1: uv (recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager. Install
it first (see the uv documentation). Then run:

```bash
uv sync --extra dev
```

This command creates `.venv` and installs the exact versions in `uv.lock`.
Use `uv run` to run a command in this environment without activation.

### Option 2: pip

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run

```bash
cd notebooks
uv run jupyter notebook lab1_hyperelasticity.ipynb
```

With pip, activate `.venv` and run `jupyter notebook` without `uv run`.

The first run downloads about 7 MB into `data/raw/`. A full run takes about
three minutes on a laptop CPU.

## Test

```bash
uv run pytest
uv run ruff check src tests
```

The tests use small synthetic files. They do not use the network.

## Data sources

1. **Soft Robotics Materials Database**: Ecoflex 00-30, uniaxial test.
   Marechal et al. (2021), *Soft Robotics* 8(3), 284-297.
   https://github.com/LucMarechal/Soft-Robotics-Materials-Database (ODbL-1.0)
2. **Treloar (1944)**: natural rubber in uniaxial, equibiaxial and pure shear
   tension. *Transactions of the Faraday Society* 40, 59-70. Digital copy from
   https://github.com/LivingMatterLab/CANN
3. **EUCLID finite element data**: plate with a hole, four load steps.
   Flaschel et al. (2021), *CMAME* 381, 113852.
   https://github.com/EUCLID-code/EUCLID-hyperelasticity (GPL-3.0)

Every URL points to a fixed git commit. The package checks each file with
its SHA-256 checksum.

## Method in short

- **Equation.** Quasi-static balance of linear momentum, $Div P = 0$, with
  $P = \frac{dW}{dF}$.
- **Variants.** Yeoh N = 1, 2, 3 (N = 1 is neo-Hookean), Yeoh with a
  volumetric term, and Ogden M = 2.
- **Graph.** DEAP stores each W as a tree. SymPy and torch receive the same
  compiled tree. networkx stores the full equation as a DAG with shared
  nodes. zss measures the tree edit distance between variants.
- **Fit.** Homogeneous tests use a relative stress residual. The field test
  uses the weak-form residual at the measured displacements.
- **Solver.** L-BFGS-B minimizes the total potential energy for the forward
  field problem.
