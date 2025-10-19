import torch.nn as nn

class LinearRegression(nn.Module):
    """Simple Linear Regression model using PyTorch."""
    
    def __init__(self, input_features: int):
        """Initialize the model.
        
        Args:
            input_features (int): Number of input features
        """
        super().__init__()
        self.linear = nn.Linear(input_features, 1)
    
    def forward(self, x):
        """Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_features)
            
        Returns:
            torch.Tensor: Predicted values
        """
        return self.linear(x)
