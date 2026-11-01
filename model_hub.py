from pathlib import Path
import importlib
from models.BaseModel import BaseModel
from loguru import logger


def _import_module(module_name: str):
    model_classes_dict = {}
    try:
        module = importlib.import_module(module_name)  # try importing the module
    except ImportError as e:
        logger.error(f"Failed to import {module_name}: {e}")
        return None
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (
            isinstance(attr, type)
            and issubclass(attr, BaseModel)
            and getattr(attr, "ENABLED", False)
        ):
            model_classes_dict[attr.__name__] = attr
    return model_classes_dict


def gather_models(path: Path) -> list[type[BaseModel]]:
    model_classes_dict = {}

    for file in path.glob("*.py"):
        if file.stem in ["__init__", "BaseModel"]:  # skip init and base model files
            continue
        module_name = f"models.{file.stem}"
        import_result = _import_module(module_name)
        if import_result is not None:
            model_classes_dict.update(import_result)

    return list(model_classes_dict.values())


def _is_defined_in_source(path: Path, class_name: str) -> bool:
    # Read source code directly
    try:
        with open(path, 'r') as f:
            source = f.read()
        return f"class {class_name}" in source
    except Exception as e:
        logger.error(f"Failed to read {path}: {e}")
        return False

def load_model(selector: str) -> BaseModel:
    # Attempt to load the model class by its selector
    for file in Path("models").glob("*.py"):
        if file.stem in ["__init__", "BaseModel"]:
            continue
        is_defined = _is_defined_in_source(file, selector)
        if is_defined:
            model_classes_dict = _import_module(f"models.{file.stem}")
            if model_classes_dict and selector in model_classes_dict:
                return model_classes_dict[selector]()

    raise ValueError(f"Model with selector '{selector}' not found")
