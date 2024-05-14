from typing import Any, Literal, Optional
from training.core.dataset import UrbanSoundDatasetCreator
from torch.optim import Adam
from torch.nn import NLLLoss
from ninja import Schema

# no layer params

# hardcode non tunable params into schema
class AudioParams(Schema):
    name: str
    problem_type: Literal["CLASSIFICATION"]
    default: UrbanSoundDatasetCreator
    criterion: NLLLoss
    optimizer_name: Adam 
    shuffle: bool
    epochs: int
    test_size: float
    batch_size: int
    # TODO add user_arch
    # user_arch: 