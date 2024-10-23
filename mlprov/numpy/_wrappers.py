import numpy
from numpy import ndarray as orig_ndarray


class ndarray(orig_ndarray):
    """
    A wrapper for numpy ndarrays to store our additional annotations.
    See https://docs.scipy.org/doc/numpy-1.13.0/user/basics.subclassing.html
    """

    def __new__(cls, input_array, provenance=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = numpy.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        if provenance is not None:
            obj.provenance = provenance
        else:
            # TODO: Generate provenance for new datasource
            raise NotImplementedError("TODO")
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        # https://numpy.org/devdocs/user/basics.subclassing.html#simple-example-adding-an-extra-attribute-to-ndarray
        if obj is None:
            return
        self.provenance = getattr(obj, "provenance", None)

    def ravel(self, order="C"):
        result = super().ravel(order)
        assert isinstance(result, ndarray)
        result.provenance = (
            self.provenance
        )  # pylint: disable=protected-access
        return result
