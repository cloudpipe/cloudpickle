import typing


def local_submod_function():
    return "hello from a file located in a locally-importable subfolder!"


class LocalSubmodClass:
    def method(self):
        return "hello from a class located in a locally-importable subfolder!"


LocalSubmodT = typing.TypeVar("LocalSubmodT")
