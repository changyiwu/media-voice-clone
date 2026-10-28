from sklearn._unavailable import make_getattr

__getattr__ = make_getattr('sklearn.feature_extraction')
