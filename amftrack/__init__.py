import logging
import sys

log_format = "%(asctime)s-[%(levelname)s]- %(name)s:%(lineno)d -> %(message)s"

# ROOT LOGGER
# Every log goes through this logger. Set level here for general logging level.
# Levels: DEBUG, INFO, ERROR, WARNING
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=log_format)

# FILTERS
# Change loglevel here for specific logging
matplotlib_logger = logging.getLogger("matplotlib")
matplotlib_logger.setLevel(logging.WARNING)
