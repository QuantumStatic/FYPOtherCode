import pickle
from types import prepare_class
from typing import Any,  Union
from myFunctions import execute_this


class DataStorage:

    _DataStorageObj = None

    def __new__(cls, *args, **kwargs):
        if cls._DataStorageObj is None:
            cls._DataStorageObj = super().__new__(cls)
        return cls._DataStorageObj

    def __init__(self):
        self.path = r"misc_data"
        self._catalogue = []
        self._data: dict[str, Any] = {}
        self._setup()

    def _setup(self):
        self._data = self._data_storage_handler()
        self._load_catalogue()

    def _data_storage_handler(self, obj_to_store=None) -> Union[dict[str, Any], None]:
        if obj_to_store is not None:
            with open(self.path, "wb") as handle:
                pickle.dump(obj_to_store, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(self.path, 'rb') as handle:
                return pickle.load(handle)

    def store_object(self, **kwargs):
        for obj in kwargs.items():
            self._data[obj[0]] = obj[1]
            self._catalogue.append(obj[0])
        self._data_storage_handler(self._data)

    def load_object(self, *access_names):
        if len(access_names) == 1:
            return self._data.setdefault(access_names[0], None)
        else:
            return tuple(self._data.setdefault(access_name, None) for access_name in access_names)

    def _load_catalogue(self):
        for key in self._data:
            self._catalogue.append(key)

    @property
    def catalogue(self) -> list[str]:
        return sorted(self.catalogue)

    def remove(self, *names):
        for name in names:
            del self._data[name]

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]

    def __contains__(self, key):
        return key in self._data

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return f"{self.__class__.__name__}({self._data})"

    def __str__(self):
        return self.__repr__()

    def __call__(self, *args, **kwargs):
        return self._data(*args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def __del__(self):
        pass


@execute_this
def main():
    storage_manager = DataStorage()
    print(storage_manager)
