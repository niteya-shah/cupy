def _ureduce(a, axis, keepdims):
    """
    Internal Function which converts the input array into
    the required shape as specified by the axis and keepdims.
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
            elif ax % a.ndim in temp_list:
                raise(ValueError("repeated axis {} specified"
                                 .format(ax % a.ndim)))
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
