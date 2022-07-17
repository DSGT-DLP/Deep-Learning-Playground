import torch
from constants import SAVED_MODEL

model = torch.load(SAVED_MODEL)
model.eval()

print(model.state_dict())
