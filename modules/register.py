from collections import defaultdict

class Registry:
    def __init__(self, name):
        self._name = name
        self._registry = {}

    def register(self, name=None):
        def decorator(cls):
            key = name or cls.__name__
            if key in self._registry:
                raise ValueError(f"{key} already registered in {self._name}")
            self._registry[key] = cls
            return cls
        return decorator

    def get(self, name):
        if name not in self._registry:
            raise KeyError(f"{name} not found in {self._name}")
        return self._registry[name]

    def list(self):
        return list(self._registry.keys())

CADB_BLOCKS=Registry('CrossAttnDownBlock2D')