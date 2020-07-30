from localvglobal.models.models import DenseNet3


_default_models = {
    'densenet10': lambda num_classes: DenseNet3(depth=10, num_classes=num_classes)
}


def default_model(name, num_classes):
    return _default_models[name.lower()](num_classes)

