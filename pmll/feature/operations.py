# -*- coding: utf-8 -*-
"""Operations for Features. Methods are patched methods for sympy.core.Expr"""
import sympy

from ..features import Feature, FeatureLin

__author__ = "Kirill Pavlov"
__email__ = "kirill.pavlov@phystech.edu"


def _decoreate_lin_feature(f):
    """Decorate method for FeatureLin input/output."""
    def wrapper(*args, **kwargs):
        is_feature_in_args = any([isinstance(a, Feature) for a in args]) or \
            any([isinstance(v, Feature) for v in kwargs.values()])

        if is_feature_in_args:
            args = [a.formula if isinstance(a, Feature) else a for a in args]
            kwargs = {k: v.formula if isinstance(v, Feature) else v
                      for k, v in kwargs.items()}
            result = f(*args, **kwargs)
            feature = FeatureLin(str(result))
            feature.formula = result
            return feature
        else:
            return f(*args, **kwargs)
    return wrapper
