"""Convenience logging helpers that play nicely with progress bars.

The module silences overly verbose loggers commonly produced by plotting
and HDF5 libraries and installs a :class:`TqdmLoggingHandler` that
writes log messages using ``tqdm.write`` so logs do not corrupt
progress-bar output.
"""

import logging
from tqdm.auto import tqdm

# silence very noisy debug logs
logging.getLogger('matplotlib.pyplot').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.getLogger('matplotlib.ticker').setLevel(logging.WARNING)
logging.getLogger('matplotlib.colorbar').setLevel(logging.WARNING)
logging.getLogger("concurrent.futures").setLevel(logging.WARNING)
logging.getLogger("h5py._conv").setLevel(logging.WARNING)
logging.getLogger("h5py._utils").setLevel(logging.WARNING)


# setup more friendly progress bar logging
class TqdmLoggingHandler(logging.Handler):
    """Logging handler that writes messages via :func:`tqdm.write`.

    Using this handler prevents logging messages from overwriting or
    corrupting tqdm progress bar output in interactive consoles.
    """
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


# Set up TqdmLoggingHandler as the default handler for the root logger
handler = TqdmLoggingHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s - %(message)s', datefmt='%H:%M:%S')
handler.setFormatter(formatter)

root_logger = logging.getLogger()
root_logger.handlers = [handler]
