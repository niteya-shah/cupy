def _ureduce(a, axis, keepdims):
    # TODO Fix helper doc

    """
    Internal Function.
    Call `func` with `a` as first argument swapping the axes to use extended
    axis on functions that don't support it natively.
    Returns result and a.shape with axis dims set to 1.
    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    func : callable
        Reduction function capable of receiving a single axis argument.
        It is called with `a` as first argument followed by `kwargs`.
    kwargs : keyword arguments
        additional keyword arguments to pass to `func`.
    Returns
    -------
    result : tuple
        Result of func(a, **kwargs) and a.shape with axis dims set to 1
        which can be used to reshape the result to the same shape a ufunc with
        keepdims=True would produce.
    """
    keepdim = None
    if isinstance(axis, int):
        ap = a
        axis = axis,

    if keepdims:
        if axis is None:
            keepdim = (1,) * a.ndim
        else:
            keepdim = list(a.shape)
            for ax in axis:
                keepdim[ax % a.ndim] = 1
            keepdim = tuple(keepdim)

    # Copy a since we need it sorted but without modifying the original array
    if axis is None:
        ap = a.flatten()
        nkeep = 0
        axis = -1
    else:
        # Reduce axes from a and put them last
        ap = a
        temp_list = list()
        for ax in axis:
            if ax >= ap.ndim or ax < -ap.ndim:
                raise(ValueError("Axis value specified {} out of dimension"
                                 " for array dimensions {}"
                                 .format(ax, a.ndim)))
            else:
                temp_list.append(ax % a.ndim)
        axis = tuple(temp_list)
        keep = set(range(a.ndim)) - set(axis)
        nkeep = len(keep)
        for i, s in enumerate(sorted(keep)):
            ap = ap.swapaxes(i, s)
        ap = ap.reshape(ap.shape[:nkeep] + (-1,)).copy()
        axis = -1
    return ap, keepdim, axis
