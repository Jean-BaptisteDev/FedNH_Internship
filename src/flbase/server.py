

from ..utils import autoassign, save_to_pkl, access_last_added_element, calculate_model_size
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader
from collections import OrderedDict


class Server:
    def __init__(self, server_config, clients_dict, **kwargs):
        """
        Initializes the server instance for federated learning, including the server configuration, clients, and server-side client.

        Parameters:
        - server_config (dict): Configuration for the server, including dataset and model specifications.
        - clients_dict (dict): Dictionary of client instances participating in federated learning.
        - kwargs (dict): Additional parameters required for the server-side client initialization, such as criterion, train/test sets, and device settings.
        """
        autoassign(locals())
        self.server_model_state_dict = None
        self.server_model_state_dict_best_so_far = None
        self.num_clients = len(self.clients_dict)
        self.strategy = None
        self.average_train_loss_dict = {}
        self.average_train_acc_dict = {}
        # global model performance
        self.gfl_test_loss_dict = {}
        self.gfl_test_acc_dict = {}
        # local model performance (averaged across all clients)
        self.average_pfl_test_loss_dict = {}
        self.average_pfl_test_acc_dict = {}
        self.active_clients_indicies = None
        self.rounds = 0
        # create a fake client on the server side; use for testing the performance of the global model
        # trainset is only used for creating the label distribution
        self.server_side_client = kwargs['client_cstr'](
            kwargs['server_side_criterion'],
            kwargs['global_trainset'],
            kwargs['global_testset'],
            kwargs['server_side_client_config'],
            -1,
            kwargs['server_side_client_device'],
            **kwargs)

    def select_clients(self, ratio):
        """
        Selects a subset of clients based on a specified participation ratio.

        Parameters:
        - ratio (float): Ratio of clients to select (between 0 and 1).

        Returns:
        - selected_indices (list): List of indices of selected clients.
        """
        assert ratio > 0.0, "Invalid ratio. Possibly the server_config['participate_ratio'] is wrong."
        num_clients = int(ratio * self.num_clients)
        selected_indices = np.random.choice(range(self.num_clients), num_clients, replace=False)
        return selected_indices

    def testing(self, round, active_only, **kwargs):
        """
        Runs testing on the current global model, either for all clients or only active clients in the current round.

        Parameters:
        - round (int): Current federated learning round.
        - active_only (bool): If True, test only on active clients.
        - kwargs (dict): Additional arguments for customization.

        Raises:
        - NotImplementedError: Method should be implemented in a subclass.
        """
        raise NotImplementedError

    def collect_stats(self, stage, round, active_only, **kwargs):
        """
        Collects and records performance statistics from clients.

        Parameters:
        - stage (str): Stage of training (e.g., 'train', 'test').
        - round (int): Current federated learning round.
        - active_only (bool): If True, collect stats only from active clients.
        - kwargs (dict): Additional parameters for customized stat collection.

        Raises:
        - NotImplementedError: Method should be implemented in a subclass.
        """
        raise NotImplementedError()

    def aggregate(self, client_uploads, round):
        """
        Aggregates updates from selected clients to update the global model.

        Parameters:
        - client_uploads (dict): Dictionary of model updates from clients.
        - round (int): Current federated learning round.

        Raises:
        - NotImplementedError: Method should be implemented in a subclass.
        """
        raise NotImplementedError

    def run(self):
        """
        Runs the entire federated learning process across multiple rounds.

        Raises:
        - NotImplementedError: Method should be implemented in a subclass.
        """
        raise NotImplementedError

    def save(self, filename, keep_clients_model=False):
        """
        Saves the current server instance and optionally removes client-specific data to save memory.

        Parameters:
        - filename (str): Path to save the serialized server object.
        - keep_clients_model (bool): If False, clears client models and training data to save memory.
        """
        if not keep_clients_model:
            for client in self.clients_dict.values():
                client.model = None
                client.trainloader = None
                client.trainset = None
                client.new_state_dict = None
        self.server_side_client.trainloader = None
        self.server_side_client.trainset = None
        self.server_side_client.testloader = None
        self.server_side_client.testset = None
        save_to_pkl(self, filename)

    def summary_setup(self):
        """
        Prints a summary of the federated learning setup, including server, dataset, and client configuration details.
        """
        info = "=" * 60 + "Run Summary" + "=" * 60
        info += "\nDataset:\n"
        info += f" dataset:{self.server_config['dataset']} | num_classes:{self.server_config['num_classes']}"
        partition = self.server_config['partition']
        info += f" | partition:{self.server_config['partition']}"
        if partition == 'iid-equal-size':
            info += "\n"
        elif partition in ['iid-diff-size', 'noniid-label-distribution']:
            info += f" | beta:{self.server_config['beta']}\n"
        elif partition == 'noniid-label-quantity':
            info += f" | num_classes_per_client:{self.server_config['num_classes_per_client']}\n "
        else:
            if 'shards' in partition.split('-'):
                pass
            else:
                raise ValueError(f" Invalid dataset partition strategy:{partition}!")
        info += "Server Info:\n"
        info += f" strategy:{self.server_config['strategy']} | num_clients:{self.server_config['num_clients']} | num_rounds: {self.server_config['num_rounds']}"
        info += f" | participate_ratio:{self.server_config['participate_ratio']} | drop_ratio:{self.server_config['drop_ratio']}\n"
        info += f"Clients Info:\n"
        client_config = self.clients_dict[0].client_config
        info += f" model:{client_config['model']} | num_epochs:{client_config['num_epochs']} | batch_size:{client_config['batch_size']}"
        info += f" | optimizer:{client_config['optimizer']} | inint lr:{client_config['client_lr']} | lr scheduler:{client_config['client_lr_scheduler']} | momentum: {client_config['sgd_momentum']} | weight decay: {client_config['sgd_weight_decay']}"
        print(info)
        mdict = self.server_side_client.get_params()
        print(f" {client_config['model']}: size:{calculate_model_size(mdict):.3f} MB | num params:{sum(mdict[key].nelement() for key in mdict) / 1e6: .3f} M")

    def summary_result(self):
        """
        Prints or returns a summary of the results from the federated learning process.

        Raises:
        - NotImplementedError: Method should be implemented in a subclass.
        """
        raise NotImplementedError
