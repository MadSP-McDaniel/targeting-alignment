import inspect
from functools import wraps
from typing import Union, get_args, get_origin, get_type_hints

import torch
from accelerate import find_executable_batch_size


def type_check(func):
    """
    A decorator that validates function arguments against their type hints.
    Supports Union, Optional, List, and basic types.

    Args:
        func: The function to be decorated

    Returns:
        Wrapped function that performs type checking before execution

    Raises:
        TypeError: If an argument's type doesn't match its type hint
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get type hints for the function
        hints = get_type_hints(func)

        # Get the function's parameter names
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        def check_type(value, expected_type):
            # Handle Union types
            if get_origin(expected_type) is Union:
                return any(check_type(value, t) for t in get_args(expected_type))

            # Handle Optional types (Union[type, None])
            if get_origin(expected_type) is Union and type(None) in get_args(
                expected_type
            ):
                if value is None:
                    return True
                return check_type(
                    value,
                    next(t for t in get_args(expected_type) if t is not type(None)),
                )

            # Handle List types
            if get_origin(expected_type) is list:
                if not isinstance(value, list):
                    return False
                # If list has type arguments, check each element
                if get_args(expected_type):
                    return all(
                        check_type(item, get_args(expected_type)[0]) for item in value
                    )
                return True

            # Handle basic types
            return isinstance(value, expected_type)

        # Check each argument
        for param_name, value in bound_args.arguments.items():
            if param_name in hints:
                expected_type = hints[param_name]
                if not check_type(value, expected_type):
                    raise TypeError(
                        f"Argument '{param_name}' is of type {type(value).__name__}, "
                        f"but {expected_type} was expected"
                    )

        return func(*args, **kwargs)

    return wrapper


def filter_kwargs(class_or_method, kwargs: dict):
    """
    Filter out kwargs that are not in the signature method or class init
    """
    if inspect.isclass(class_or_method):
        parameters = inspect.signature(class_or_method.__init__).parameters
    else:
        parameters = inspect.signature(class_or_method).parameters

    class_kwargs = {
        k: v for k, v in parameters.items() if v.kind in [v.KEYWORD_ONLY, v.VAR_KEYWORD]
    }
    return {k: v for k, v in kwargs.items() if k in class_kwargs}


def arange_args(class_or_method, args: dict):
    """
    Arange args in the order they are in the signature of the class or method
    """
    if inspect.isclass(class_or_method):
        parameters = inspect.signature(class_or_method.__init__).parameters
    else:
        parameters = inspect.signature(class_or_method).parameters

    class_args = {
        k: v
        for k, v in parameters.items()
        if v.kind in [v.POSITIONAL_ONLY, v.POSITIONAL_OR_KEYWORD] and k != "self"
    }
    return (args[k] for k in class_args.keys())


def get_all_args(class_or_method, args_and_kwargs: dict):
    """
    Get all arguments in the signature of the class or method
    """
    args = arange_args(class_or_method, args_and_kwargs)
    kwargs = filter_kwargs(class_or_method, args_and_kwargs)
    return args, kwargs


def silhouette_score(X, labels, loss=False, metric="euclidean"):
    if type(labels) != type(torch.HalfTensor()):
        labels = torch.HalfTensor(labels)
    if not labels.is_cuda:
        labels = labels.cuda()

    if type(X) != type(torch.HalfTensor()):
        X = torch.HalfTensor(X)
    if not X.is_cuda:
        X = X.cuda()

    unique_labels = torch.unique(labels)

    A = _intra_cluster_distances_block(
        X, labels, unique_labels, metric=metric, device=X.device
    )
    B = _nearest_cluster_distance_block(
        X, labels, unique_labels, metric=metric, device=X.device
    )
    sil_samples = (B - A) / torch.maximum(A, B)

    # nan values are for clusters of size 1, and should be 0
    mean_sil_score = torch.mean(torch.nan_to_num(sil_samples))
    if loss:
        return -mean_sil_score
    else:
        return float(mean_sil_score.cpu().numpy())


def _intra_cluster_distances_block(
    X, labels, unique_labels, metric="euclidean", device="cuda:0"
):
    intra_dist = torch.zeros(labels.size(), dtype=torch.float32, device=device)
    values = [
        _intra_cluster_distances_block_(
            X[torch.where(labels == label)[0]], metric=metric
        )
        for label in unique_labels
    ]
    for label, values_ in zip(unique_labels, values):
        intra_dist[torch.where(labels == label)[0]] = values_
    return intra_dist


def _intra_cluster_distances_block_(subX, metric="euclidean"):
    if metric == "euclidean":
        distances = torch.cdist(subX, subX)
        return distances.sum(axis=1) / (distances.shape[0] - 1)
    elif metric == "cosine":
        subX = torch.nn.functional.normalize(subX, p=2, dim=-1)
        distances = 1 - torch.matmul(subX, subX.transpose(-1, -2))
    return distances.sum(axis=1) / (distances.shape[0] - 1)


def _nearest_cluster_distance_block(
    X, labels, unique_labels, metric="euclidean", device="cuda:0"
):
    inter_dist = torch.full(
        labels.size(), torch.inf, dtype=torch.float32, device=device
    )
    label_combinations = torch.combinations(unique_labels, 2)

    values = [
        _nearest_cluster_distance_block_(
            X[torch.where(labels == label_a)[0]],
            X[torch.where(labels == label_b)[0]],
            metric=metric,
        )
        for label_a, label_b in label_combinations
    ]

    for (label_a, label_b), (values_a, values_b) in zip(label_combinations, values):

        indices_a = torch.where(labels == label_a)[0]
        inter_dist[indices_a] = torch.minimum(values_a, inter_dist[indices_a])
        del indices_a
        indices_b = torch.where(labels == label_b)[0]
        inter_dist[indices_b] = torch.minimum(values_b, inter_dist[indices_b])
        del indices_b
    return inter_dist


def _nearest_cluster_distance_block_(subX_a, subX_b, metric="euclidean"):
    if metric == "euclidean":
        dist = torch.cdist(subX_a, subX_b)
    elif metric == "cosine":
        subX_a = torch.nn.functional.normalize(subX_a, p=2, dim=-1)
        subX_b = torch.nn.functional.normalize(subX_b, p=2, dim=-1)
        dist = 1 - torch.matmul(subX_a, subX_b.transpose(-1, -2))

    dist_a = dist.mean(axis=1)
    dist_b = dist.mean(axis=0)
    return dist_a, dist_b


def savefig(path, size=[4, 3]):
    import os

    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams["figure.autolayout"] = True
    # Sane default fig size for papers
    matplotlib.rcParams["figure.figsize"] = [4, 3]

    # Uses Opentype-compatible fonts
    # conferences often require this for camera ready, so if you don't do it pre-submission you'll have a nightmare at camera-ready time.
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    # Automatically make the directory hierarchy so I can just save figures with path names
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Makes background transparent so plots can go in slides and look good
    plt.gcf().patch.set_alpha(0)
    # Default fig size
    plt.gcf().set_size_inches(*size)
    # Make figure fill whole PDF (otherwise figs have huge margins in LaTeX
    plt.tight_layout(
        pad=0,
    )
    plt.savefig(path, bbox_inches="tight")
    plt.clf()

    # Sets seaborn whitegrid on every plot for consistency (darkgrid is nice for slides)
    sns.set_style("whitegrid")


MODELS = [
    "llama2",
    "qwen2",
    "gemma1",
    "gemma2",
    "granite",
    "llama3",
    "mistral",
    "zephyrrmu",
]

MODELS_MAP = {
    "llama2": "Llama 2",
    "qwen2": "Qwen 2.5",
    "gemma1": "Gemma 1",
    "gemma2": "Gemma 2",
    "granite": "Granite",
    "llama3": "Llama 3",
    "mistral": "Mistral",
    "zephyrrmu": "Zephyr RMU",
}

REVERSE_MODELS_MAP = {v: k for k, v in MODELS_MAP.items()}

LAYER_MAP = {
    "llama2": 32,
    "qwen2": 28,
    "gemma1": 28,
    "gemma2": 42,
    "granite": 40,
    "llama3": 32,
    "mistral": 32,
    "zephyrrmu": 32,
}

METRIC_MAP = {
    "agreement_llm": "Accuracy",
    "agreement_llm_f1": "F1",
    "cross_source_agreement_llm_f1": "Cross-Source F1",
}

SOURCE_MAP = {
    "advbench": "AdvBench",
    "or-bench": "OR-Bench",
}

MARKER_MAP = {
    "llama2": "o",
    "qwen2": "s",
    "gemma1": "v",
    "gemma2": "^",
    "granite": "<",
    "llama3": ">",
    "mistral": "D",
    "zephyrrmu": "P",
}

DATASET_LINESTYLES = {"AdvBench": "solid", "OR-Bench": "dotted"}

DATASET_MAP = {
    "advbench_harmful.json": "AdvBench",
    "advbench_harmless.json": "AdvBench",
    "or-bench_harmful.json": "OR-Bench",
    "or-bench_harmless.json": "OR-Bench",
}
