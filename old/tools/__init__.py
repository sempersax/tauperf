import os
import logging

log = logging.getLogger('tools')
if not os.environ.get('DEBUG', False):
    log.setLevel(logging.INFO)
