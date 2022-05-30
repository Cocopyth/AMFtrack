import logging
import sys

log_format = "%(asctime)s-[%(levelname)s]- %(name)s:%(lineno)d -> %(message)s"

# ROOT LOGGER
# Every log goes through this logger. Set level here for general logging level.
# Levels: DEBUG, INFO, ERROR, WARNING
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)

# FILTERS
# Change loglevel here for specific logging
matplotlib_logger = logging.getLogger("matplotlib")
matplotlib_logger.setLevel(logging.WARNING)
matplotlib_logger = logging.getLogger("amftrack.util.dbx")
matplotlib_logger.setLevel(logging.WARNING)
# Removed
for logger_name in ["PIL.TiffImagePlugin"]:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.WARNING)
