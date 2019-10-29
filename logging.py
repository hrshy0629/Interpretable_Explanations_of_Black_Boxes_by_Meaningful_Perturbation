# -*- coding: utf-8 -*-
import logging
LEVEL = {
    'notest': logging.NOTSET,
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL,
         }###Set log level
class Logger:
    def __init__(self):
        self._level = logging.INFO
        self._formatter = logging.Formatter(
            '%(asctime)s %(levelname)s %(message)s')
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel((self._level))
    def __getattr__(self, attrname):#attrname is name of attribute which is not in the class Logger.
        return getattr(self._logger, attrname)#The function can define Logger's name
    @property#The flag make function can be called like an attribute. But It does not give function a value.
    def level(self):
        return self._level
    @level.setter##It can give the function a value.
    def level(self, lev):
        if lev in LEVEL.keys():
            self._level = LEVEL[lev]
            self._logger.setLevel(self._level)
        else:
            raise ValueError(f'the {lev} level is not a correct logging level!')

    @property
    def formatter(self):
        return self._formatter
    @formatter.setter
    def formatter(self, fmt):
        if isinstance(fmt, logging.Formatter):
            self._formatter = fmt
        else:
            raise ValueError(f'the {fmt} formatter is not a Formatter object!')

    def addCmdHandler(self):
        cmd_handler = logging.StreamHandler()
        cmd_handler.setLevel(self._level)
        cmd_handler.setFormatter(self._formatter)
        self._logger.addHandler(cmd_handler)
    def addFileHandler(self, path = 'Logging.log'):
        file_handler = logging.FileHandler(path)
        file_handler.setLevel(self._level)
        file_handler.setFormatter(self._formatter)
        self._logger.addHandler(file_handler)

logger = Logger()##instantiation