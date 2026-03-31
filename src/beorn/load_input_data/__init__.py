"""Module that handles the import of nbody simulation data from various formats."""
from .nbody_base import NBodyLoader as NBodyLoader
from .cosmo_sim_artificial import *  # noqa: F403
from .cosmo_sim_pkdgrav import *  # noqa: F403
from .cosmo_sim_thesan import *  # noqa: F403
from .cosmo_sim_py21cmfast import *  # noqa: F403
