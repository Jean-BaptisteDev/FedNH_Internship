from ..model import Model  
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.models as models
from .ResNet import ResNet18, ResNet18NoNorm


class Conv2Cifar(Model):
    """Convolutional Neural Network model designed for CIFAR dataset classification."""
    
    def __init__(self, config):
        super().__init__(config)
        # Define the first convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
        # Define the second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        # Define max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layer definitions
        self.linear1 = nn.Linear(64 * 5 * 5, 384)  # Input size derived from the convolutional output
        self.linear2 = nn.Linear(384, 192)
        # Prototype layer for output class logits, bias term removed for fair comparison
        self.prototype = nn.Linear(192, config['num_classes'], bias=False)

    def forward(self, x):
        """Forward pass through the network."""
        # Pass input through the first convolutional layer followed by ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Pass through the second convolutional layer followed by ReLU and pooling
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the output for the fully connected layers
        x = x.view(-1, 64 * 5 * 5)
        # Pass through the first fully connected layer
        x = F.relu(self.linear1(x))
        # Pass through the second fully connected layer
        x = F.relu(self.linear2(x))
        # Compute logits using the prototype layer
        logits = self.prototype(x)
        return logits

    def get_embedding(self, x):
        """Return feature embeddings along with logits."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        logits = self.prototype(x)
        return x, logits


class Conv2CifarNH(Model):
    """Convolutional Neural Network with Normalized Hybrid Prototypes for CIFAR classification."""
    
    def __init__(self, config):
        super().__init__(config)
        # Determine if embeddings should be returned based on configuration
        self.return_embedding = config['FedNH_return_embedding']
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear1 = nn.Linear(64 * 5 * 5, 384)
        self.linear2 = nn.Linear(384, 192)
        # Initialize prototype weights from the configuration
        temp = nn.Linear(192, config['num_classes'], bias=False).state_dict()['weight']
        self.prototype = nn.Parameter(temp)
        # Scaling parameter for logits
        self.scaling = torch.nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        """Forward pass with normalization of embeddings and prototypes."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.linear1(x))
        feature_embedding = F.relu(self.linear2(x))
        
        # Normalize feature embeddings
        feature_embedding_norm = torch.norm(feature_embedding, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        feature_embedding = torch.div(feature_embedding, feature_embedding_norm)
        
        # Normalize prototypes if they require gradients
        if self.prototype.requires_grad == False:
            normalized_prototype = self.prototype
        else:
            prototype_norm = torch.norm(self.prototype, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            normalized_prototype = torch.div(self.prototype, prototype_norm)
        
        # Compute logits as a dot product of embeddings and normalized prototypes
        logits = torch.matmul(feature_embedding, normalized_prototype.T)
        logits = self.scaling * logits

        if self.return_embedding:
            return feature_embedding, logits
        else:
            return logits


class ResNetMod(Model):
    """Modified ResNet model for classification tasks."""
    
    def __init__(self, config):
        super().__init__(config)
        # Choose ResNet architecture based on normalization configuration
        if config['no_norm']:
            self.backbone = ResNet18NoNorm(num_classes=config['num_classes'])
        else:
            self.backbone = ResNet18(num_classes=config['num_classes'])
        # Prototype layer for output logits
        self.prototype = nn.Linear(self.backbone.linear.in_features, config['num_classes'], bias=False)
        self.backbone.linear = None  # Remove the linear layer from ResNet

    def forward(self, x):
        """Forward pass through the modified ResNet."""
        feature_embedding = self.backbone(x)  # Get feature embeddings from ResNet backbone
        logits = self.prototype(feature_embedding)  # Compute logits using the prototype layer
        return logits

    def get_embedding(self, x):
        """Return feature embeddings along with logits."""
        feature_embedding = self.backbone(x)
        logits = self.prototype(feature_embedding)
        return feature_embedding, logits


class ResNetModNH(Model):
    """ResNet model with Normalized Hybrid Prototypes for CIFAR classification."""
    
    def __init__(self, config):
        super().__init__(config)
        self.return_embedding = config['FedNH_return_embedding']
        # Choose ResNet architecture based on normalization configuration
        if config['no_norm']:
            self.backbone = ResNet18NoNorm(num_classes=config['num_classes'])
        else:
            self.backbone = ResNet18(num_classes=config['num_classes'])
        # Initialize prototype weights from the ResNet model
        temp = nn.Linear(self.backbone.linear.in_features, config['num_classes'], bias=False).state_dict()['weight']
        self.prototype = nn.Parameter(temp)
        self.backbone.linear = None  # Remove the linear layer from ResNet
        self.scaling = torch.nn.Parameter(torch.tensor([20.0]))  # Scaling factor for logits
        self.activation = None  # Placeholder for storing activation

    def forward(self, x):
        """Forward pass with normalization of embeddings and prototypes."""
        feature_embedding = self.backbone(x)  # Get feature embeddings
        feature_embedding_norm = torch.norm(feature_embedding, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        feature_embedding = torch.div(feature_embedding, feature_embedding_norm)

        # Normalize prototypes if they require gradients
        if self.prototype.requires_grad == False:
            normalized_prototype = self.prototype
        else:
            prototype_norm = torch.norm(self.prototype, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            normalized_prototype = torch.div(self.prototype, prototype_norm)

        # Compute logits as a dot product of embeddings and normalized prototypes
        logits = torch.matmul(feature_embedding, normalized_prototype.T)
        logits = self.scaling * logits
        self.activation = self.backbone.activation  # Store the activation for potential later use
        
        if self.return_embedding:
            return feature_embedding, logits
        else:
            return logits


class Conv2MRINH(nn.Module):
    def __init__(self, config):
        super(Conv2MRINH, self).__init__()
        self.return_embedding = config.get('FedNH_return_embedding', False)
        self.in_channels = 1  # Assuming grayscale MRI images; change if needed
        self.num_classes = config['num_classes']

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Compute the size of the feature map after convolutions and pooling
        self._to_linear = None
        self._init_conv2d()

        # Fully connected layers
        self.linear1 = nn.Linear(self._to_linear, 384)  # Input size adjusted
        self.linear2 = nn.Linear(384, 192)

        # Prototype layer
        self.prototype = nn.Parameter(torch.randn(192, self.num_classes))  # Initialize prototypes randomly
        self.scaling = torch.nn.Parameter(torch.tensor([1.0]))

    def _init_conv2d(self):
        """Calculate the size of the feature map after convolutions and pooling."""
        with torch.no_grad():
            x = torch.zeros(1, self.in_channels, 224, 224)  # Sample input tensor
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            self._to_linear = x.numel()  # Store the number of features to flatten

    def forward(self, x):
        """Forward pass for the Conv2MRINH model."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self._to_linear)  # Flatten the output
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))

        # Compute logits using the prototype layer
        logits = torch.matmul(x, self.prototype)
        logits = self.scaling * logits  # Scale logits

        if self.return_embedding:
            return x, logits  # Return both embedding and logits
        else:
            return logits
