import hydra


def instantiate(cfg, *args, **kwargs):
    return hydra.utils.instantiate(cfg, *args, **kwargs)