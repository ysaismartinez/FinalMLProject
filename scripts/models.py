import torch.nn as nn
# I was struggling mightily with the imports
# So I went back to my notes from Alfredo's boot camp last sumnmer and segregated the logic below into its own script for clarity and modularity
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.model(x)

