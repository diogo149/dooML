import random
import json
from os import path, system
from utils import try_mkdir


class Trial(object):
    DIRECTORY = "trial"

    def __init__(self, description, **kwargs):
        self._trial = dict(**kwargs)
        self['description'] = description
        while True:
            self['id'] = hash(random.random())
            if not path.exists(self._filename()):
                system("touch {}".format(self._filename()))
                break

    def __getattr__(self, name):
        return self[name]

    def __getitem__(self, name):
        return self._trial[name]

    def __setitem__(self, name, value):
        assert isinstance(name, str)
        self(**{name: value})

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self._close()

    def __repr__(self):
        return """Trial(**{})""".format(self._trial.__repr__())

    def __call__(self, **kwargs):
        self._trial.update(**kwargs)

    def _filename(self):
        try_mkdir(Trial.DIRECTORY)
        return path.join(Trial.DIRECTORY, "{}.json".format(self.id))

    def _close(self):
        with open(self._filename(), 'w') as outfile:
            json.dump(self._trial, outfile)
