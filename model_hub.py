from pathlib import Path
import importlib
from models.BaseModel import BaseModel
from loguru import logger


def gather_models(path: Path) -> list[type[BaseModel]]:
    model_classes = []

    for file in path.glob("*.py"):
        if file.stem in ["__init__", "BaseModel"]:  # skip init and base model files
            continue
        module_name = f"models.{file.stem}"
        try:
            module = importlib.import_module(module_name)  # try importing the module
        except ImportError as e:
            logger.error(f"Failed to import {module_name}: {e}")
            continue
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, BaseModel)
                and getattr(attr, "ENABLED", False)
            ):
                model_classes.append(attr)

    return model_classes


def load_model(selector: str) -> BaseModel:
    all_models = gather_models(Path("models"))
    for model_class in all_models:
        if model_class.__name__ == selector:
            return model_class()

    raise ValueError(f"Model with selector '{selector}' not found")
