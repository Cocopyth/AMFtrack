from functools import wraps
from time import time
import logging


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        logging.info(f"Function {f.__name__} took {te-ts:2.4f} seconds")
        return result

    return wrap
