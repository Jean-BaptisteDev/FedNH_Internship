from collections import OrderedDict, Counter  # Importing necessary data structures
import numpy as np  # Importing NumPy for numerical operations
from tqdm import tqdm  # Importing tqdm for progress bars
from copy import deepcopy  # Importing deepcopy for copying objects
import torch  # Importing PyTorch for deep learning
try:
    import wandb  # Importing Weights & Biases for experiment tracking
except ModuleNotFoundError:
    pass  # If wandb is not installed, continue without it
from ..server import Server  # Importing the Server class from the parent module
from ..client import Client  # Importing the Client class from the parent module
from ..models.CNN import *  # Importing all CNN models from the models module
from ..models.MLP import *  # Importing all MLP models from the models module
from ..utils import setup_optimizer, linear_combination_state_dict, setup_seed  # Importing utility functions
from ..strategies.FedAvg import FedAvgServer  # Importing FedAvgServer for federated learning strategy
from ...utils import autoassign, save_to_pkl, access_last_added_element  # Importing utility functions from sibling module
import math  # Importing math module for mathematical functions
import time  # Importing time module for time-related functions

class FedUHClient(Client):
    """
    Client class for federated learning using the FedUH strategy.
    Inherits from the base Client class.
    """
    def __init__(self, criterion, trainset, testset, client_config, cid, device, **kwargs):
        """
        Initializes the FedUHClient instance.
        
        Parameters:
        - criterion: Loss function
        - trainset: Training dataset
        - testset: Testing dataset
        - client_config: Configuration dictionary for the client
        - cid: Client ID
        - device: Device to run the model on (CPU or GPU)
        - kwargs: Additional keyword arguments
        """
        super().__init__(criterion, trainset, testset, client_config, cid, device, **kwargs)
        self._initialize_model()  # Initializes the model

    @staticmethod
    def _get_orthonormal_basis(m, n):
        """
        Generates an orthonormal basis matrix of shape (m, n).
        
        Parameters:
        - m: Number of rows
        - n: Number of columns
        
        Returns:
        - W: Orthonormal basis matrix
        """
        W = torch.rand(m, n)  # Initialize a random matrix
        # Apply Gram-Schmidt process to create orthonormal vectors
        for i in range(m):
            q = W[i, :]  # Select the ith row
            for j in range(i):
                q = q - torch.dot(W[j, :], W[i, :]) * W[j, :]  # Orthogonalize
            if torch.equal(q, torch.zeros_like(q)):  # Check linear independence
                raise ValueError("The row vectors are not linearly independent!")
            q = q / torch.sqrt(torch.dot(q, q))  # Normalize the vector
            W[i, :] = q  # Update the matrix
        return W  # Return the orthonormal basis

    def _initialize_model(self):
        """Initializes the model based on the configuration provided."""
        # Parse the model from the config file
        self.model = eval(f"{self.client_config['model']}NH")(self.client_config).to(self.device)
        # Move criterion to the appropriate device if it has stateful tensors
        self.criterion = self.criterion.to(self.device)
        try:
            self.model.prototype.requires_grad_(False)  # Set the prototype parameters to not require gradients
            if self.client_config['FedNH_head_init'] == 'orthogonal':
                # Initialize the prototype with orthogonal weights
                m, n = self.model.prototype.shape
                self.model.prototype.data = torch.nn.init.orthogonal_(torch.rand(m, n)).to(self.device)
            elif self.client_config['FedNH_head_init'] == 'uniform' and self.client_config['dim'] == 2:
                # Initialize prototypes in a uniform distribution on a circle
                r = 1.0  # Radius
                num_cls = self.client_config['num_classes']
                W = torch.zeros(num_cls, 2)  # Initialize a tensor for storing the prototype weights
                for i in range(num_cls):
                    theta = i * 2 * torch.pi / num_cls  # Calculate angle for each class
                    W[i, :] = torch.tensor([r * math.cos(theta), r * math.sin(theta)])  # Set prototype position
                self.model.prototype.copy_(W)  # Copy the prototype positions
            else:
                raise NotImplementedError(f"{self.client_config['FedNH_head_init']} + {self.client_config['num_classes']}d")
        except AttributeError:
            raise NotImplementedError("Only support linear layers now.")
        if self.client_config['FedNH_fix_scaling'] == True:
            # Fix scaling parameter if specified
            self.model.scaling.requires_grad_(False)  # Freeze the scaling parameter
            self.model.scaling.data = torch.tensor(30.0).to(self.device)  # Set initial scaling value
            print('self.model.scaling.data:', self.model.scaling.data)  # Output the scaling value

    def training(self, round, num_epochs):
        """
        Trains the model for a specified number of epochs.
        
        Parameters:
        - round: Current round of training
        - num_epochs: Number of epochs to train
        """
        setup_seed(round + self.client_config['global_seed'])  # Set random seed for reproducibility
        self.model.train()  # Set model to training mode
        self.num_rounds_particiapted += 1  # Increment the number of rounds participated
        loss_seq = []  # Initialize loss sequence list
        acc_seq = []  # Initialize accuracy sequence list
        
        if self.trainloader is None:
            raise ValueError("No trainloader is provided!")  # Raise error if trainloader is not provided
        
        optimizer = setup_optimizer(self.model, self.client_config, round)  # Set up the optimizer
        
        # Start training
        for i in range(num_epochs):
            epoch_loss, correct = 0.0, 0  # Initialize epoch loss and correct predictions
            for _, (x, y) in enumerate(self.trainloader):
                # Forward pass
                x, y = x.to(self.device), y.to(self.device)  # Move data to the specified device
                yhat = self.model.forward(x)  # Get predictions from the model
                loss = self.criterion(yhat, y)  # Calculate loss

                # Backward pass
                self.model.zero_grad(set_to_none=True)  # Clear previous gradients
                loss.backward()  # Backpropagate the loss
                torch.nn.utils.clip_grad_norm_(parameters=filter(lambda p: p.requires_grad, self.model.parameters()), max_norm=10)  # Clip gradients
                optimizer.step()  # Update model parameters
                
                # Stats
                predicted = yhat.data.max(1)[1]  # Get the predicted class
                correct += predicted.eq(y.data).sum().item()  # Count correct predictions
                epoch_loss += loss.item() * x.shape[0]  # Rescale loss to batch size

            epoch_loss /= len(self.trainloader.dataset)  # Average loss over the dataset
            epoch_accuracy = correct / len(self.trainloader.dataset)  # Calculate accuracy
            loss_seq.append(epoch_loss)  # Append loss for this epoch
            acc_seq.append(epoch_accuracy)  # Append accuracy for this epoch
        
        self.new_state_dict = self.model.state_dict()  # Save the current model state
        self.train_loss_dict[round] = loss_seq  # Record training loss for this round
        self.train_acc_dict[round] = acc_seq  # Record training accuracy for this round

    def upload(self):
        """Uploads the model parameters to the server."""
        return self.new_state_dict  # Return the current state dictionary

    def testing(self, round, testloader=None):
        """
        Tests the model on the provided testloader or the default testloader.
        
        Parameters:
        - round: Current round of testing
        - testloader: Optional test loader
        """
        self.model.eval()  # Set model to evaluation mode
        if testloader is None:
            testloader = self.testloader
        test_count_per_class = Counter(testloader.dataset.targets.numpy())
        # all_classes_sorted = sorted(test_count_per_class.keys())
        # test_count_per_class = torch.tensor([test_count_per_class[cls] * 1.0 for cls in all_classes_sorted])
        # num_classes = len(all_classes_sorted)
        num_classes = self.client_config['num_classes']
        test_count_per_class = torch.tensor([test_count_per_class[cls] * 1.0 for cls in range(num_classes)])
        test_correct_per_class = torch.tensor([0] * num_classes)

        weight_per_class_dict = {'uniform': torch.tensor([1.0] * num_classes),
                                 'validclass': torch.tensor([0.0] * num_classes),
                                 'labeldist': torch.tensor([0.0] * num_classes)}
        for cls in self.label_dist.keys():
            weight_per_class_dict['labeldist'][cls] = self.label_dist[cls]
            weight_per_class_dict['validclass'][cls] = 1.0
        # start testing
        with torch.no_grad():
            for i, (x, y) in enumerate(testloader):
                # forward pass
                x, y = x.to(self.device), y.to(self.device)
                yhat = self.model.forward(x)
                # stats
                predicted = yhat.data.max(1)[1]
                classes_shown_in_this_batch = torch.unique(y).cpu().numpy()
                for cls in classes_shown_in_this_batch:
                    test_correct_per_class[cls] += ((predicted == y) * (y == cls)).sum().item()
        acc_by_critertia_dict = {}
        for k in weight_per_class_dict.keys():
            acc_by_critertia_dict[k] = (((weight_per_class_dict[k] * test_correct_per_class).sum()) /
                                        ((weight_per_class_dict[k] * test_count_per_class).sum())).item()

        self.test_acc_dict[round] = {'acc_by_criteria': acc_by_critertia_dict,
                                     'correct_per_class': test_correct_per_class,
                                     'weight_per_class': weight_per_class_dict}


class FedUHServer(FedAvgServer):
    """
    Server class for federated learning using the FedUH strategy.
    Inherits from the FedAvgServer class.
    """

    def __init__(self, server_config, clients_dict, exclude, **kwargs):
        """
        Initializes the FedUHServer instance.
        
        Parameters:
        - server_config: Configuration dictionary for the server
        - clients_dict: Dictionary of clients connected to the server
        - exclude: List of layer keys to exclude from aggregation
        - kwargs: Additional keyword arguments for flexibility
        """
        super().__init__(server_config, clients_dict, exclude, **kwargs)  # Call the parent constructor to initialize base attributes
        
        # Check if there are any layers to exclude from aggregation
        if len(self.exclude_layer_keys) > 0:
            print(f"FedUHServer: the following keys will not be aggregated:\n", self.exclude_layer_keys)  # Print excluded layer keys
        
        freeze_layers = []  # Initialize a list to track layers that will not be updated
        # Iterate through named parameters of the server's client model
        for param in self.server_side_client.model.named_parameters():
            # Check if the parameter requires gradients (i.e., is trainable)
            if param[1].requires_grad == False:
                freeze_layers.append(param[0])  # Add the parameter name to the freeze_layers list
        
        # If there are any frozen layers, print them out
        if len(freeze_layers) > 0:
            print("FedUHServer: the following layers will not be updated:", freeze_layers)  # Notify which layers won't be updated
