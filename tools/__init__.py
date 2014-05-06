import os
import logging

log = logging.getLogger('cmd')
if not os.environ.get('DEBUG', False):
    log.setLevel(logging.INFO)
