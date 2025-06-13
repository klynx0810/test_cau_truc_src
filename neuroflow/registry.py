from .src.losses.mse import MSELoss
from .src.losses.categorical_crossentropy import CategoricalCrossentropy
from .src.losses.binary_crossentropy import BinaryCrossentropy

from .src.optimizers.sgd import SGD
from .src.optimizers.adam import Adam

from .src.layers.activations.relu import ReLU
from .src.layers.activations.sigmoid import Sigmoid
from .src.layers.activations.tanh import Tanh
from .src.layers.activations.softmax import Softmax

from .src.metrics.accuracy import Accuracy

LOSS_REGISTRY = {
    "mse": MSELoss,
    "categorical_crossentropy": CategoricalCrossentropy,
    "binary_crossentropy": BinaryCrossentropy,
}

OPTIMIZER_REGISTRY = {
    "sgd": SGD,
    "adam": Adam,
}

ACTIVATION_REGISTRY = {
    "relu": ReLU,
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "softmax": Softmax,
}

METRICS_REGISTRY = {
    "accuracy": Accuracy,
}

def get_loss(identifier):
    if isinstance(identifier, str):
        return LOSS_REGISTRY[identifier.lower()]()
    return identifier

def get_optimizer(identifier, **kwargs):
    if isinstance(identifier, str):
        opt_cls = OPTIMIZER_REGISTRY.get(identifier.lower())
        if opt_cls is None:
            raise ValueError(f"Unknown optimizer '{identifier}'")
        return opt_cls(**kwargs)
    return identifier 

def get_activation(identifier):
    if isinstance(identifier, str):
        act_cls = ACTIVATION_REGISTRY.get(identifier.lower())
        if act_cls is None:
            raise ValueError(f"Unknown activation '{identifier}'")
        return act_cls()
    return identifier

def get_metrics(identifier):
    if isinstance(identifier, str):
        act_cls = METRICS_REGISTRY.get(identifier.lower())
        if act_cls is None:
            raise ValueError(f"Unknown metrics '{identifier}'")
        return act_cls()
    return identifier