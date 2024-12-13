import numpy as np
import torch
import matplotlib.pyplot as plt

###################
# Utility Functions
###################

def visualize_augmentations(dataset, samples=36):
    import matplotlib.pyplot as plt
    import random
    
    plt.figure(figsize=(10, 10))
    for i in range(samples):
        idx = random.randint(0, len(dataset)-1)
        data = dataset[idx][0]
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        if data.shape[0] == 1:  # If channels first, move to last
            data = np.transpose(data, (1, 2, 0))
        plt.subplot(6, 6, i + 1)
        plt.imshow(data.squeeze(), cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()


