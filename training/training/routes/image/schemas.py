from typing import Any, Literal, Optional
from ninja import Schema


class LayerParams(Schema):
    value: str
    parameters: list[Any]


class ImageParams(Schema):
    name: str
    problem_type: Literal["CLASSIFICATION"]
    default: Optional[str]
    criterion: str
    optimizer_name: str
    shuffle: bool
    epochs: int
    test_size: float
    batch_size: int
    user_arch: list[LayerParams]
