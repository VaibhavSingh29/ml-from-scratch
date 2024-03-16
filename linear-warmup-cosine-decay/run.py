import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import math

class RandomDataset(Dataset):
    """
    Define a dummy dataset
    """
    def __init__(self, size: int, num_features: int) -> None:
        super().__init__()
        
        self.size = size
        self.num_features = num_features
        self.data = torch.randn(self.size, self.num_features)
        self.labels = torch.zeros(self.size, dtype=torch.long)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx: int):
        return self.data[idx], self.labels[idx]


def set_seed(seed: int = 42):
    torch.manual_seed(seed)

def plot(total_training_steps: int, lr_updates: list):
    plt.xlabel('Training step')
    plt.ylabel('Learning rate')
    plt.plot(range(total_training_steps), lr_updates)
    plt.show()

def main():

    set_seed()

    DATA_SIZE = 1000
    NUM_FEATURES = 10
    BATCH_SIZE = 32
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_EPOCHS = 100
    INITIAL_LR = 1e-4
    PEAK_LR = 1e-2

    dataset = RandomDataset(size=DATA_SIZE, num_features=NUM_FEATURES)
    trainloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = nn.Sequential(
        nn.Linear(NUM_FEATURES, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, weight_decay=0.1)

    print(f"Total number of training steps = {NUM_EPOCHS * len(trainloader)}")

    WARMUP_STEPS = 0.1 * NUM_EPOCHS * len(trainloader) # 10% of total number of training steps
    min_lr = 0.1 * INITIAL_LR
    delta = (PEAK_LR - INITIAL_LR) / WARMUP_STEPS
    global_step = -1
    lr_updates = []

    for epoch in range(NUM_EPOCHS):
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            global_step += 1

            if global_step < WARMUP_STEPS:
                lr = INITIAL_LR + global_step * delta
            else:
                progress = (global_step - WARMUP_STEPS) / (NUM_EPOCHS * len(trainloader) - WARMUP_STEPS)
                lr = min_lr + (PEAK_LR - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            lr_updates.append(optimizer.param_groups[0]['lr'])

            '''
            Compute loss and update parameters hereafter
            '''
        
    plot(NUM_EPOCHS * len(trainloader), lr_updates)


if __name__ == "__main__":
    main()