import os
import logging

log = logging.getLogger('skim')
if not os.environ.get('DEBUG', False):
    log.setLevel(logging.INFO)
