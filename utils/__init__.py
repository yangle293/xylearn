import theano
import numpy

np = numpy


def assignName(variable, anon="anonymous_variable"):
    """
    If variable has a name, returns that name.
    Otherwise, returns anon
    """

    if hasattr(variable, 'name') and variable.name is not None:
        return variable.name

    return anon


def toSharedX(value, name=None, borrow=False):
    """Transform value into a shared variable of type floatX"""
    return theano.shared(theano._asarray(value, dtype=theano.config.floatX),
                         name=name,
                         borrow=borrow)


def toFloatX(variable):
    """Casts a given variable into dtype config.floatX
    numpy ndarrays will remain numpy ndarrays
    python floats will become 0-D ndarrays
    all other types will be treated as theano tensors"""

    if isinstance(variable, float):
        return numpy.cast[theano.config.floatX](variable)

    if isinstance(variable, numpy.ndarray):
        return numpy.cast[theano.config.floatX](variable)

    return theano.tensor.cast(variable, theano.config.floatX)


def getConstantX(value):
    """
        Returns a constant of value `value` with floatX dtype
    """
    return theano.tensor.constant(numpy.asarray(value,
                                                dtype=theano.config.floatX))


def subDict(d, keys):
    """ Create a subdictionary of d with the keys in keys """
    result = {}
    for key in keys:
        if key in d: result[key] = d[key]
    return result


def safeUpdate(dict_to, dict_from):
    """
    Like dict_to.update(dict_from), except don't overwrite any keys.
    """
    for key, val in dict(dict_from).iteritems():
        if key in dict_to:
            raise KeyError(key)
        dict_to[key] = val
    return dict_to


def safeUnion(a, b):
    """
    Does the logic of a union operation without the non-deterministic
    ordering of python sets
    """
    if not isinstance(a, list):
        raise TypeError("Expected first argument to be a list, but got " + str(type(a)))
    assert isinstance(b, list)
    c = []
    for x in a + b:
        if x not in c:
            c.append(x)
    return c