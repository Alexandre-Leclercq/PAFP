from omegaconf import OmegaConf

def if_not_null(a, b):
    return a if a is not None else b

RESOLVERS = {
    "if_not_null": if_not_null,
}

def register_all_resolvers():
    for name, func in RESOLVERS.items():
        OmegaConf.register_new_resolver(name, func, replace=True)