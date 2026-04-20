# BEoRN: Bubbles during the Epoch Of Reionization Numerical-simulator

[![License](https://img.shields.io/github/license/cosmic-reionization/BEoRN.svg)](https://github.com/cosmic-reionization/BEoRN/blob/master/LICENSE.md)
[![GitHub Repository](https://img.shields.io/github/repo-size/cosmic-reionization/BEoRN)](https://github.com/cosmic-reionization/BEoRN)
[![CI status](https://github.com/cosmic-reionization/BEoRN/actions/workflows/test-install.yaml/badge.svg)](https://github.com/cosmic-reionization/BEoRN/actions/workflows/test-install.yaml)
[![Documentation](https://img.shields.io/badge/Documentation-here-blue)](https://cosmic-reionization.github.io/BEoRN)
[![PyPI version](https://badge.fury.io/py/beorn.svg)](https://badge.fury.io/py/beorn)

BEoRN is a simulation tool designed to model the state of the intergalactic medium (IGM) during cosmic dawn and reionization ([Schaeffer, Giri & Schneider 2023](https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.2942S/abstract)). It leverages one-dimensional radiative transfer calculations to efficiently model the temperature evolution of intergalactic gas and growth of ionized regions (bubbles) around early galaxies embedded in dark matter halos (e.g., [Schneider, Giri & Mirocha 2021](https://ui.adsabs.harvard.edu/abs/2021PhRvD.103h3025S/abstract); [Schneider, Schaeffer & Giri 2023](https://ui.adsabs.harvard.edu/abs/2023PhRvD.108d3030S/abstract)).

BEoRN is designed to be **flexible**, **user-friendly**, and **fast**, allowing researchers to efficiently explore various astrophysical scenarios and their impact on the 21-cm signal from neutral hydrogen.

**BEoRN is actively developed.** We welcome feedback and contributions. If you encounter any issues, please inform the developers by opening a [GitHub issue](https://github.com/cosmic-reionization/BEoRN/issues).

## Key Features

* 📦 **Lightweight & Modular:** A Python package suitable for simulation modules and analysis tools.
* 📊 **Data Visualization:** Utilities for assembling time/coeval cube data for easy visualization.
* 🔄  **Reproducible:** Testing and CI-ready structure to support reproducible development.
* 🌌 **Flexible Inputs:** Natively reads halo catalogs from simulations such as [Thesan](https://thesan-project.com/) and [PkdGrav](https://bitbucket.org/dpotter/pkdgrav3), or generates synthetic catalogs on the fly relying on [21cmFAST](https://github.com/21cmfast/21cmFAST).

## Documentation

Full documentation is available at: **[https://cosmic-reionization.github.io/BEoRN](https://cosmic-reionization.github.io/BEoRN)**

## Installation

### Standard Installation
You can install BEoRN directly from GitHub using `pip`:
```bash
pip install git+https://github.com/cosmic-reionization/beorn.git
```

### With Optional Dependencies
The `extra` option installs additional packages needed for generating synthetic halo catalogs with `21cmFAST` and for comparing halo mass functions against analytical models (`hmf`):
```bash
pip install "git+https://github.com/cosmic-reionization/beorn.git[extra]"
```
**Note:** If `21cmFAST` installation fails, please refer to the [21cmFAST](https://github.com/21cmfast/21cmFAST) repository and install it manually first, then install BEoRN without the extra option.

### Development Installation
For a local, editable installation (useful if you want to modify the code):

1. Clone the repository:
```bash
git clone https://github.com/cosmic-reionization/beorn.git
cd beorn
```
2. Install in editable mode:
```bash
pip install -e .
```
To also install the optional dependencies:
```bash
pip install -e ".[extra]"
```

### Verifying the Installation
After installing, you can run the unit tests to verify everything works correctly:
```bash
pip install ".[dev]"
python -m pytest tests -v
```

## Dependencies
The core dependencies are listed in `pyproject.toml` and include [numpy](https://numpy.org/), [scipy](https://scipy.org/), [h5py](https://www.h5py.org/), [mpi4py](https://mpi4py.readthedocs.io/en/stable/), [astropy](https://www.astropy.org/), [matplotlib](https://matplotlib.org/), and [tools21cm](https://github.com/sambit-giri/tools21cm).

Optional extras (`numba`, `pylians`, `torch`, `jax`) enable faster or GPU-accelerated particle-to-mesh mapping backends but are not required for standard use.

## Project Layout
- `src/beorn/`: Package source code.
- `docs/`: Documentation source.
- `examples/`: Runnable examples and Jupyter notebooks to get started quickly.

For the stochastic `f_st` workflow, see `examples/full_run_fstar.py` together with the companion documentation file `examples/full_run_fstar_file.rst`.

**Note**: This repository focuses on code and workflows. Heavy simulation outputs (coeval/temporal cubes, large data products) are expected to be stored externally due to size.

## 📖 Citation
If you use this package in your research, please consider citing the following paper:

```bibtex
@article{Schaeffer_2023,
    title={beorn: a fast and flexible framework to simulate the epoch of reionization and cosmic dawn},
    volume={526},
    ISSN={1365-2966},
    url={[http://dx.doi.org/10.1093/mnras/stad2937](http://dx.doi.org/10.1093/mnras/stad2937)},
    DOI={10.1093/mnras/stad2937},
    number={2},
    journal={Monthly Notices of the Royal Astronomical Society},
    publisher={Oxford University Press (OUP)},
    author={Schaeffer, Timothée and Giri, Sambit K and Schneider, Aurel},
    year={2023},
    month=sep,
    pages={2942–2959}
}
```

## 👨‍💻 Authors
- Sambit Giri - [GitHub](https://github.com/sambit-giri)
- Rémy Moll - [GitHub](https://github.com/moll-re)
- Timothée Schaeffer - [GitHub](https://github.com/timotheeschaeffer)
- Aurel Schneider - [GitHub](https://github.com/aurelschneider)

## Contributing
Contributions are welcome! If you find bugs or unexpected behavior, please open a [Github issue](https://github.com/cosmic-reionization/beorn/issues). For detailed guidelines on contributing code or setting up a development environment, please see `CONTRIBUTING.rst`.

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/cosmic-reionization/BEoRN/blob/main/LICENSE) file for details.