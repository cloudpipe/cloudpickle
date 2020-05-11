import sys


if sys.version_info.major == 3 and sys.version_info.minor < 8:
    try:
        import pickle5 as pickle
        import pickle5._pickle as _pickle
    except ImportError:
        import pickle
        import _pickle
else:
    import pickle
    import _pickle
