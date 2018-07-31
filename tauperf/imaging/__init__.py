
import logging


FORMAT = "{color}{levelname}:{name}] {message}"
class CustomFormatter(logging.Formatter):
    def __init__(self, fmt=FORMAT, datefmt=None):
        logging.Formatter.__init__(self, fmt, datefmt)

    def format(self, record):
        if not hasattr(record, "message"):
            record.message = record.getMessage()
        record.asctime = self.formatTime(record, self.datefmt)
        return self._fmt.format(color="", **record.__dict__)

log = logging.getLogger()
log.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(CustomFormatter())
log.addHandler(handler)

#from .. import log; log = log[__name__]
# from .processing import process_taus
