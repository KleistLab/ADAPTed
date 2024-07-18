"""
ADAPT (Adapter and poly(A) Detection And Profiling Tool)

Copyright (c) 2023 by Wiep K. van der Toorn
Contact: w.vandertoorn@fu-berlin.de

"""

import pprint
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Tuple, Type, TypeVar, Union, get_args

import numpy as np
import toml


def get_field_type_annotation(field):
    type_hint = field.type
    if hasattr(type_hint, "__origin__") and type_hint.__origin__ is Union:
        args = get_args(type_hint)
        if type(None) in args:
            # This is an Optional type
            return next(arg for arg in args if arg is not type(None))
    return type_hint


@dataclass
class BaseConfig:
    def pretty_print(self):
        return pprint.pformat(asdict(self), sort_dicts=False)

    def dict(self):
        return asdict(self)

    def typed_dict(self):
        typed_dict = {}

        for attr, value in self.dict().items():
            from typing import get_type_hints

            attr_type = get_type_hints(type(self)).get(attr, None)

            if attr_type is Tuple[float, float]:
                typed_value = (
                    tuple(value) if isinstance(value, (list, np.ndarray)) else value
                )
                if attr.endswith("_range") and len(typed_value) == 2:
                    v_new = [-np.inf, np.inf]
                    if typed_value[0] is not None:
                        v_new[0] = float(typed_value[0])
                    if typed_value[1] is not None:
                        v_new[1] = float(typed_value[1])
                    typed_value = v_new
            elif attr_type is bool:
                typed_value = bool(value)
            elif attr_type is int:
                typed_value = int(value)
            elif attr_type is float:
                typed_value = float(value)
            elif attr_type is str:
                typed_value = str(value)
            else:
                typed_value = value

            typed_dict[attr] = typed_value
        return typed_dict

    def to_toml(self, file_path: str):
        with open(file_path, "w") as toml_file:
            toml.dump(self.typed_dict(), toml_file)


@dataclass
class NestedConfig(BaseConfig):
    def pretty_print(self):
        for key, val in self.dict().items():
            if isinstance(val, BaseConfig):
                print(f"{key}:\n{val.pretty_print()}")
            else:
                print(f"{key}: {val}")

    def typed_dict(self):
        return {
            k: (v if not isinstance(v, BaseConfig) else v.typed_dict())
            for k, v in self.dict().items()
        }


T = TypeVar("T", bound=NestedConfig)


def load_nested_config_from_file(
    file_path: Union[str, Path], config_class: Type[T] = NestedConfig
) -> T:
    config_file = toml.load(file_path)
    config_file_keys = set(config_file.keys())

    valid_keys = set([f.name for f in fields(config_class)])
    unknown_keys = [key for key in config_file_keys if key not in valid_keys]

    if unknown_keys:
        raise ValueError(
            f"Invalid config file. Unknown key(s): {', '.join(unknown_keys)}. "
            f"Valid keys are: {', '.join(valid_keys)}"
        )

    config_obj = config_class()

    # First, handle entries without a section header
    for key in config_file_keys:
        if not isinstance(config_file[key], dict):
            if key in valid_keys:
                setattr(config_obj, key, config_file[key])
            else:
                raise ValueError(f"Invalid config file. Unknown key: {key}")

    # Then, handle sections
    for section, content in config_file.items():
        if isinstance(content, dict):
            if section in valid_keys:
                field_type = next(
                    get_field_type_annotation(f)
                    for f in fields(config_class)
                    if f.name == section
                )
                if issubclass(field_type, BaseConfig):
                    pass
                    try:
                        setattr(config_obj, section, field_type(**content))
                    except:
                        raise ValueError(
                            f"Invalid config file. Could not parse section {section}"
                            f" with content {content}"
                            f" as {field_type}"
                        )
                else:
                    raise ValueError(
                        f"Invalid section type for {section}: {field_type}"
                    )
            else:
                raise ValueError(f"Invalid config file. Unknown section: {section}")

    return config_obj
