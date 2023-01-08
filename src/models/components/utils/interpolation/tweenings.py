import pytweening


def interpolate(alpha, method):
    out = vars(pytweening)[method](alpha)
    return out
