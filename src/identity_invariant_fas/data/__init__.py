"""Lazy public exports keep standalone media qualification independent of torch."""
from importlib import import_module

_MODULES = {
    'FASSample': 'manifest', 'ManifestDataset': 'manifest',
    'read_manifest': 'manifest', 'write_manifest': 'manifest',
    'load_nuaa_samples': 'nuaa', 'subject_kfold_split': 'splits',
    'subject_train_val_split': 'splits', 'build_eval_transform': 'transforms',
    'build_train_transform': 'transforms',
}


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(name)
    value = getattr(import_module('.' + _MODULES[name], __name__), name)
    globals()[name] = value
    return value

__all__ = [
    "FASSample",
    "ManifestDataset",
    "read_manifest",
    "write_manifest",
    "load_nuaa_samples",
    "subject_kfold_split",
    "subject_train_val_split",
    "build_eval_transform",
    "build_train_transform",
]
