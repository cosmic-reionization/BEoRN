# BEoRN

**Bubbles during the Epoch Of Reionization Numerical-simulator** (BEORN) is a simulation-tool that models the state of the intergalactic medium (IGM) during cosmic dawn and reionization. More information can be found in the [Documentation](https://cosmic-reionization.github.io/BEoRN).

Key features:
- Light, modular python package suitable for simulation modules and analysis tools.
- Utilities for assembling time / coeval cube data for easy visualization.
- Testing and CI-ready structure to support reproducible development.



### CONTRIBUTING

If you find any bugs or unexpected behavior in the code, please feel free to open a [Github issue](https://github.com/cosmic-reionization/beorn/issues).


Project layout
- `src/beorn/`- package source
- `docs/` - documentation source (built site lives at the docs link)
- `docs/examples` - runnable examples and notebooks

Notes about data and simulations
- This repository focuses on code and workflows; heavy simulation outputs (coeval/temporal cubes, large data products) are expected to be stored externally due to size.
