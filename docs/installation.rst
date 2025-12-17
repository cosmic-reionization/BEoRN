============
Installation
============

Beorn can be installed using `pip`. This should ensure that all required dependencies are made available during the installation process. Note that some dependencies may require additional system packages to be installed.

The dependencies for Beorn are listed in the `pyproject.toml` file located at the root of the project directory.

.. literalinclude:: ../pyproject.toml
   :language: toml
   :start-after: # include in docs
   :end-before: # end include in docs
   :caption: Dependencies from pyproject.toml


Finally, to install Beorn, run the following command::

    pip install git+https://github.com/cosmic-reionization/beorn.git


A local, editable installation is also possible using pip (or equivalent). This is useful if you want to modify the code and test your changes without having to reinstall the package each time.

1. Download or clone the `Repository on GitHub <https://github.com/cosmic-reionization/beorn>`_

2. Install in editable mode by running the following command::

    pip install -e /path/to/beorn




Additional Dependencies
========================
The outputs of beorn are strongly linked to the input given by halo catalogs. Beorn can natively read halo catalogs from well-known simulation projects like the `Thesan <https://thesan-project.com/>`_ and `PkdGrav <https://arxiv.org/abs/1609.08621>`_ simulations.

Additionally, beorn can leverage semi-numerical simulation suites to generate synthetic halo catalogs on the fly. Currently, beorn supports the `21cmFAST <https://github.com/21cmfast/21cmFAST>`_ suite but an additional installation step is required to enable this feature. To install beorn with 21cmFAST support, run the following command::

    pip install git+https://github.com/cosmic-reionization/beorn.git[extra]

or directly install 21cmFAST using pip::

    pip install 21cmfast

