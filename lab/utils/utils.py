import importlib

def instantiate_from_class_path(class_path: str, kwargs: dict):
    # Split module and class
    module_path, class_name = class_path.rsplit(".", 1)
    # Import the module
    module = importlib.import_module(module_path)
    # Get the class
    cls = getattr(module, class_name)
    # Instantiate with kwargs
    return cls(**kwargs)