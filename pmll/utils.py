class _classproperty(property):

    """ Implement property behaviour for classes.

    class A():

        @_classproperty
        @classmethod
        def name(cls):
            return cls.__name__

    """

    def __get__(self, obj, type_):
        return self.fget.__get__(None, type_)()


def _cached(f):
    ''' Decorator that makes a method cached.'''

    attr_name = '_cached_' + f.__name__

    def wrapper(obj, *args, **kwargs):
        if not hasattr(obj, attr_name):
            setattr(obj, attr_name, f(obj, *args, **kwargs))
        return getattr(obj, attr_name)
    return wrapper


classproperty = lambda f: _classproperty(classmethod(f))
cached_property = lambda f: property(_cached(f))
cached_classproperty = lambda f: classproperty(_cached(f))