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
    """
    A logging handler that uses tqdm to display log messages in the console. This way, logs written using this handler do not interfere with the tqdm progress bar output.
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
