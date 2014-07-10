# -*- coding: utf-8 -*-
"""Operations for Features. Methods are patched methods for sympy.core.Expr"""
import sympy


def wrap_feature(feature_scale):
    """Decorate method for Features with given scale."""
    def decorate_function(f):
        def wrapper(*args, **kwargs):
            from ..feature import Feature

            fargs = [
                feature for feature in list(args) + list(kwargs.values())
                if isinstance(feature, Feature)
            ]

            if fargs:
                args = tuple(a.formula if isinstance(a, Feature) else a
                             for a in args)
                kwargs = {k: v.formula if isinstance(v, Feature) else v
                          for k, v in kwargs.items()}

                result = f(*args, **kwargs)
                feature = Feature(formula=result, scale=feature_scale).proxy
                feature._atoms_map.update(dict([
                    (k, v) for arg in fargs for k, v in arg._atoms_map.items()
                ]))
                return feature
            else:
                return f(*args, **kwargs)
        return wrapper
    return decorate_function

And = wrap_feature("bin")(sympy.And)
Xor = wrap_feature("bin")(sympy.Xor)
Or = wrap_feature("bin")(sympy.Or)

Add = wrap_feature("lin")(sympy.Add)
Mul = wrap_feature("lin")(sympy.Mul)
Pow = wrap_feature("lin")(sympy.Pow)
Inverse = wrap_feature("lin")(sympy.Inverse)

sin = wrap_feature("lin")(sympy.sin)
cos = wrap_feature("lin")(sympy.cos)
tan = wrap_feature("lin")(sympy.tan)
asin = wrap_feature("lin")(sympy.asin)
acos = wrap_feature("lin")(sympy.acos)
atan = wrap_feature("lin")(sympy.atan)
