import logging
import os
import sys

from src.utils_.utils import root_path


class MyFormatter(logging.Formatter):
    err_fmt = f"%(asctime)s %(levelname)s %(filename)s @function %(funcName)s line %(lineno)s - %(message)s"
    dbg_fmt = f"%(asctime)s %(levelname)s %(name)s line %(lineno)s - %(message)s"
    info_fmt = f"%(asctime)s %(levelname)s %(name)s %(message)s"

    def __init__(self):
        super().__init__(fmt="%(levelno)d: %(msg)s", datefmt="%H:%M:%S", style='%')

    def format(self, record):

        # Save the original format configured by the user
        # when the logger formatter was instantiated
        format_orig = self._style._fmt

        # Replace the original format with one customized by logging level
        if record.levelno == logging.DEBUG:
            self._style._fmt = MyFormatter.dbg_fmt

        elif record.levelno == logging.INFO:
            self._style._fmt = MyFormatter.info_fmt

        elif record.levelno == logging.ERROR:
            self._style._fmt = MyFormatter.err_fmt

        # Call the original formatter class to do the grunt work
        result = logging.Formatter.format(self, record)

        # Restore the original format configured by the user
        self._style._fmt = format_orig

        return result


def convert_str2level(level_str):
    """
    The numeric values of logging levels are given in the following [website](https://docs.python.org/3.11/library/logging.html#logging-levels)
    :param level_str: string of logging level
    :return: native value of logging level
    """
    if 'DEBUG' == level_str.upper().strip():
        return logging.DEBUG
    elif 'INFO' == level_str.upper().strip():
        return logging.INFO
    elif 'WARNING' == level_str.upper().strip():
        return logging.WARNING
    elif 'ERROR' == level_str.upper().strip():
        return logging.ERROR
    elif 'CRITICAL' == level_str.upper().strip():
        return logging.CRITICAL
    elif 'NOTSET' == level_str.upper().strip():
        return logging.NOTSET
    else:
        raise ValueError(f'Unknown logging level {level_str}')


def setup_logger(name, log_file=None, level: str = 'DEBUG'):
    """
    Setup logger
    :param name: the name of module
    :param log_file: the path of log file including file name
    :param level: ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "NOTSET"], default: "DEBUG", case insensitive
    :return:
    """
    logger = logging.getLogger(name)
    logger.setLevel(level=convert_str2level(level))
    if not logger.handlers:
        # create the handlers and call logger.addHandler(logging_handler)
        logStreamFormatter = logging.Formatter(
            fmt=f"%(levelname)-8s %(asctime)s \t %(filename)s @function %(funcName)s line %(lineno)s - %(message)s",
            datefmt="%H:%M:%S"
        )

        fmt = MyFormatter()
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(fmt)
        console_handler.setLevel(level=logging.DEBUG)

        logger.addHandler(console_handler)

        logFileFormatter = logging.Formatter(
            fmt=f"%(levelname)s %(asctime)s (%(relativeCreated)d) \t %(filename)s F%(funcName)s L%(lineno)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        save_path_dir = os.path.join(root_path(), 'logs') if log_file is None else log_file
        if not os.path.exists(save_path_dir) or len(save_path_dir) == 0:
            os.makedirs(save_path_dir)
        fileHandler = logging.FileHandler(filename=os.path.join(save_path_dir, 'test.log'))
        fileHandler.setFormatter(logFileFormatter)
        fileHandler.setLevel(level=logging.ERROR)

        logger.addHandler(fileHandler)
        logger.propagate = False
        return logger
