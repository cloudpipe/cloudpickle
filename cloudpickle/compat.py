import sys


if sys.version_info.major == 3 and sys.version_info.minor < 8:
    try:
        import pickle5 as pickle  # noqa: F401
        import pickle5._pickle as _pickle  # noqa: F401
        from pickle5._pickle import Pickler  # noqa: F401
    except ImportError:
        import pickle  # noqa: F401
        from pickle import _Pickler as Pickler  # noqa: F401
else:
    import pickle  # noqa: F401
    from _pickle import Pickler  # noqa: F401
