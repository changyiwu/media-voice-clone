# librosa 0.9.2 在 import 時需要的子模組，見 pyproject.toml
import importlib

from ._unavailable import make_getattr

__version__ = '0.0.0+unavailable'
_SUBMODULES = ('decomposition', 'cluster', 'feature_extraction', 'neighbors')
_unavailable = make_getattr('sklearn')


def __getattr__(name):
    # from sklearn import cluster 會先 hasattr(sklearn, 'cluster')，而 hasattr 只吞 AttributeError，
    # 所以子模組要在這裡載入，不能落到 ImportError
    if name in _SUBMODULES:
        return importlib.import_module(f'{__name__}.{name}')
    return _unavailable(name)
