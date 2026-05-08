import torch
import numpy as np
from model import SimpleModel
from omegaconf import OmegaConf

# 1. Load the weights from your previously saved .npz file
# (Or use the dictionary you created in memory)
data = np.load('model_weights.npz')
print(data)

# 2. Initialize your PyTorch model
config = OmegaConf.load("configs/model/large_ecod.yaml")
model = SimpleModel(config)

# 3. Map the names and apply the Transpose
# Recall: Linear weight in PT is (out, in), TF is (in, out)
with torch.no_grad():
    # Layer 0 (First Linear)
    model.model[0].weight.copy_(
        torch.from_numpy(data['layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE']).t()
    )
    model.model[0].bias.copy_(
        torch.from_numpy(data['layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE'])
    )

    # Layer 2 (Second Linear - index 1 is Tanh)
    model.model[2].weight.copy_(
        torch.from_numpy(data['layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE']).t()
    )
    model.model[2].bias.copy_(
        torch.from_numpy(data['layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE'])
    )

# 4. Save the PyTorch state_dict
torch.save(model.state_dict(), 'converted_pytorch_model.pth')
