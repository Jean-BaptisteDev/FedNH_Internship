import torch
from collections import OrderedDict
import sys
sys.path.append("../../")
from copy import deepcopy
from src.utils import sampler, sampler_reuse, DatasetSplit, setup_seed
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

"""
Configure Optimizer
"""

def setup_optimizer(model, config, round):
    """
    Configures the optimizer for the client model based on the provided configuration and current training round.

    Parameters:
    - model: The client model for which the optimizer is being set up.
    - config: Configuration dictionary containing optimizer settings.
    - round: Current training round, used to adjust the learning rate if necessary.

    Returns:
    - optimizer: The configured optimizer for the client model.
    """
    # Determine learning rate based on the specified learning rate scheduler
    if config['client_lr_scheduler'] == 'stepwise':
        # Decrease learning rate after halfway through the training rounds
        if round < config['num_rounds'] // 2:
            lr = config['client_lr']
        else:
            lr = config['client_lr'] * 0.1

    elif config['client_lr_scheduler'] == 'diminishing':
        # Apply a diminishing learning rate based on the specified decay rate
        lr = config['client_lr'] * (config['lr_decay_per_round'] ** (round - 1))
    else:
        raise ValueError('unknown client_lr_scheduler')
    
    # Set up the optimizer based on the specified type
    if config['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            momentum=config['sgd_momentum'],
            weight_decay=config['sgd_weight_decay']
        )
    elif config['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=1e-5
        )
    elif config['optimizer'] == 'RMSprop':
        optimizer = torch.optim.RMSprop(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            alpha=config['rmsprop_alpha'],
            eps=1e-08,
            weight_decay=config['rmsprop_weight_decay'],
            momentum=config['rmsprop_momentum']
        )
    else:
        raise ValueError(f"Unknown optimizer {config['optimizer']}")
    
    return optimizer

"""
client initialization
"""

def setup_clients(Client, trainset, testset, criterion, client_config_lst, device, **kwargs):
    """
    Initializes clients with their respective training and testing datasets.

    Parameters:
    - Client: The class used to create client instances.
    - trainset: The overall training dataset.
    - testset: The overall testing dataset (can be None).
    - criterion: Loss function used for training.
    - client_config_lst: List of configurations for each client.
    - device: The device (CPU/GPU) to run the model on.
    - **kwargs: Additional arguments, including server configurations.

    Returns:
    - all_clients_dict: Dictionary containing initialized client instances.
    """
    num_clients = kwargs['server_config']['num_clients']
    partition = kwargs['server_config']['partition']
    num_classes = kwargs['server_config']['num_classes']

    assert len(client_config_lst) == num_clients, "Inconsistent num_clients and len(client_config_lst)."

    # Handle non-IID data partitioning
    if 'noniid' == partition[:6]:
        trainset_per_client_dict, stats_dict = sampler(trainset, num_clients, partition, ylabels=trainset.targets,
                                                       num_classes=num_classes, **kwargs)
        if testset is None:
            testset_per_client_dict = {cid: None for cid in range(num_clients)}
        else:
            if kwargs['same_testset']:
                testset_per_client_dict = {cid: testset for cid in range(num_clients)}
            else:
                testset_per_client_dict = sampler_reuse(testset, stats_dict, ylabels=testset.targets,
                                                        num_classes=num_classes, **kwargs)
    else:
        # Handle IID data partitioning
        trainset_per_client_dict, stats_dict = sampler(trainset, num_clients, partition, num_classes=num_classes, **kwargs)
        if testset is None:
            testset_per_client_dict = {cid: None for cid in range(num_clients)}
        else:
            if kwargs['same_testset']:
                testset_per_client_dict = {cid: testset for cid in range(num_clients)}
            else:
                testset_per_client_dict = sampler_reuse(testset, stats_dict, num_classes=num_classes, **kwargs)

    all_clients_dict = {}
    for cid in range(num_clients):
        # Ensure each client has a unique model with the same initial weights
        setup_seed(2022)  # For reproducibility
        all_clients_dict[cid] = Client(
            criterion,
            trainset_per_client_dict[cid],
            testset_per_client_dict[cid],
            client_config_lst[cid],
            cid,
            device,
            **kwargs
        )
    return all_clients_dict

def create_clients_from_existing_ones(Client, clients_dict, newtrainset, increment, criterion, **kwargs):
    """
    Creates new clients based on existing ones while maintaining the same data distribution.

    Parameters:
    - Client: The class used to create client instances.
    - clients_dict: Dictionary of existing client instances.
    - newtrainset: The new training dataset to use for creating clients.
    - increment: Increment value used to generate new training indices.
    - criterion: Loss function used for training.
    - **kwargs: Additional arguments, including configurations for client creation.

    Returns:
    - all_clients_dict: Dictionary containing newly created client instances.
    """
    num_clients = len(clients_dict)
    all_clients_dict = {}

    same_pool = kwargs.get('same_pool', False)  # Check if clients share the same data pool
    scale = kwargs.get('scale', len(newtrainset) // increment - 1)  # Default scale calculation

    for cid in range(num_clients):
        client = clients_dict[cid]
        data_idxs = client.trainset.idxs  # Get indices of current client's training data
        add_idxs = []

        if same_pool:
            for cls in client.count_by_class.keys():
                num_sample_cls = client.count_by_class[cls]
                target_num_sample_cls = min(num_sample_cls * scale, len(newtrainset.get_fake_imgs_idxs(cls)))
                add_idxs += np.random.choice(newtrainset.get_fake_imgs_idxs(cls), target_num_sample_cls, replace=False).tolist()
        else:
            for i in data_idxs:
                for j in range(scale):
                    add_idxs.append(i + increment * (j + 1))  # Generate new indices

        full_idxs = data_idxs + add_idxs  # Combine existing and new indices
        client_newtrainset = DatasetSplit(newtrainset, full_idxs)  # Create a new dataset for the client
        all_clients_dict[cid] = Client(
            criterion,
            client_newtrainset,
            client.testset,
            client.client_config,
            client.cid,
            client.group,
            client.device,
            **kwargs
        )
    return all_clients_dict

"""
resume training
"""

from copy import deepcopy
from ..utils import load_from_pkl

def resume_training(server_config, checkpoint, model):
    """
    Resumes training from a checkpoint.

    Parameters:
    - server_config: Configuration for the server.
    - checkpoint: Path to the saved checkpoint file.
    - model: The model to resume training with.

    Returns:
    - server: The server instance with restored state.
    """
    server = load_from_pkl(checkpoint)  # Load the server state from the checkpoint
    server.server_config = server_config  # Update the server configuration
    for c in server.clients_dict.values():
        c.model = deepcopy(model)  # Restore the model for each client
        c.set_params(server.server_model_state_dict)  # Load model parameters
        c.model.to(c.device)  # Move model to the appropriate device
        c.model.init()  # Re-initialize the model
    print("Resume Training")
    print(f"Rounds performed: {server.rounds}")
    return server

"""
state_dict operation
"""

def scale_state_dict(this, scale, inplace=True, exclude=set()):
    """
    Scales the state_dict of a model by a given factor.

    Parameters:
    - this: The state_dict to scale.
    - scale: Scaling factor.
    - inplace: Whether to modify the original state_dict or create a new one.
    - exclude: Set of keys to exclude from scaling.

    Returns:
    - ans: Scaled state_dict.
    """
    with torch.no_grad():
        if not inplace:
            ans = deepcopy(this)  # Create a copy if not inplace
        else:
            ans = this
        for state_key in this.keys():
            if state_key not in exclude:
                ans[state_key] = this[state_key] * scale  # Scale the parameters
        return ans

def linear_combination_state_dict(this, other, this_weight=1.0, other_weight=1.0, exclude=set()):
    """
    Combines two state_dicts linearly with given weights.

    Parameters:
    - this: First state_dict.
    - other: Second state_dict.
    - this_weight: Weight for the first state_dict.
    - other_weight: Weight for the second state_dict.
    - exclude: Set of keys to exclude from combination.

    Returns:
    - ans: Combined state_dict.
    """
    with torch.no_grad():
        ans = deepcopy(this)  # Create a new state_dict
        for state_key in this.keys():
            if state_key not in exclude:
                ans[state_key] = this[state_key] * this_weight + other[state_key] * other_weight  # Combine the states
        return ans



def average_list_of_state_dict(state_dict_lst, exclude=set()):
    """
    Averages a list of state_dicts from multiple clients, excluding specified keys.

    Parameters:
    - state_dict_lst: List of state_dicts to average.
    - exclude: Set of keys to exclude from averaging.

    Returns:
    - ans: Averaged state_dict.
    """
    assert type(state_dict_lst) == list  # Ensure the input is a list
    num_participants = len(state_dict_lst)  # Get the number of participants (clients)
    keys = state_dict_lst[0].keys()  # Retrieve keys from the first state_dict

    with torch.no_grad():  # Disable gradient tracking
        ans = OrderedDict()  # Initialize an ordered dictionary for the result
        
        # Iterate through each key in the state_dicts
        for key in keys:
            if key not in exclude:  # Check if the key should be excluded
                # Initialize the first entry of the averaged state_dict
                ans[key] = deepcopy(state_dict_lst[0][key])
                
                # Sum the state_dicts for all clients
                for idx in range(1, num_participants):
                    ans[key] += state_dict_lst[idx][key]
                
                # Average the accumulated values
                ans[key] = ans[key] / num_participants

    return ans  # Return the averaged state_dict


def weight_sum_of_dict_of_state_dict(dict_state_dict, weight_dict):
    """
    Computes a weighted sum of state_dicts from multiple clients.

    Parameters:
    - dict_state_dict: Dictionary containing state_dicts for each client.
    - weight_dict: Dictionary of weights for each client's state_dict.

    Returns:
    - ans: Weighted sum of the state_dicts.
    """
    layer_keys = next(iter(dict_state_dict.values())).keys()  # Get the keys of the layers from the first client

    with torch.no_grad():  # Disable gradient tracking
        ans = OrderedDict()  # Initialize an ordered dictionary for the result

        # Iterate over each layer in the state_dicts
        for layer in layer_keys:
            ans[layer] = None  # Initialize the layer entry in the result
            # Sum the weighted state_dicts for all clients
            for cid in dict_state_dict.keys():
                if ans[layer] is None:
                    # For the first client, create a deep copy and apply the weight
                    ans[layer] = deepcopy(dict_state_dict[cid][layer]) * weight_dict[cid]
                else:
                    # For subsequent clients, add their weighted contributions
                    ans[layer] += dict_state_dict[cid][layer] * weight_dict[cid]

    return ans  # Return the weighted sum of the state_dicts



