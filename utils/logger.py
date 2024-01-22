import logging
from tqdm import tqdm
from typing import Callable

from logging.handlers import RotatingFileHandler

# adds logger file, restrict it to X lines
log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s')
file_handler = RotatingFileHandler(
        "logs.txt",
        mode='a',
        maxBytes=1000000,
        backupCount=3,
        encoding=None,
        delay=0,
        )
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(log_formatter)

log = logging.getLogger()
log.setLevel(logging.INFO)
log.addHandler(file_handler)

colors = {
        "red": "\033[91m",
        "yellow": "\033[93m",
        "reset": "\033[0m",
        "white": "\033[0m",
        "purple": "\033[95m",
        }

def get_coloured_logger(color_asked: str) -> Callable:
    """used to print color coded logs"""
    col = colors[color_asked]

    # all logs are considered "errors" otherwise the datascience libs just
    # overwhelm the logs
    def printer(string: str, **args) -> str:
        inp = string
        if isinstance(string, dict):
            try:
                string = rtoml.dumps(string, pretty=True)
            except Exception:
                string = json.dumps(string, indent=2)
        if isinstance(string, list):
            try:
                string = ",".join(string)
            except:
                pass
        try:
            string = str(string)
        except:
            try:
                string = string.__str__()
            except:
                string = string.__repr__()
        log.info(string)
        tqdm.write(col + string + colors["reset"], **args)
        return inp
    return printer


whi = get_coloured_logger("white")
yel = get_coloured_logger("yellow")
red = get_coloured_logger("red")
