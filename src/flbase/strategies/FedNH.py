from collections import OrderedDict, Counter
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import torch
try:
    import wandb
except ModuleNotFoundError:
    pass
from ..server import Server
from ..client import Client
from ..models.CNN import *
from ..models.MLP import *
from ..utils import setup_optimizer, linear_combination_state_dict, setup_seed
from ...utils import autoassign, save_to_pkl, access_last_added_element
from .FedUH import FedUHClient, FedUHServer
import math
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import os
import plotly.express as px
from scipy.spatial import ConvexHull
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances

class FedNHClient(FedUHClient):
    """
    Client class for the Federated NH (FedNH) algorithm.

    This class represents an individual client in a Federated Learning setup. 
    It includes methods for prototype estimation based on class distributions and 
    optional advanced aggregation using density-based weighting for better prototype accuracy.
    """
    
    def __init__(self, criterion, trainset, testset, client_config, cid, device, **kwargs):
        """
        Initializes a FedNH client.

        Args:
            criterion (torch.nn.Module): Loss function criterion for training.
            trainset (torch.utils.data.Dataset): Training dataset for the client.
            testset (torch.utils.data.Dataset): Test dataset for the client.
            client_config (dict): Configuration dictionary for client parameters.
            cid (int): Client ID.
            device (torch.device): Device (CPU or GPU) where the client operates.
            **kwargs: Additional keyword arguments.

        Attributes:
            count_by_class_full (torch.Tensor): Full count of samples per class for prototype estimation.
        """
        super().__init__(criterion, trainset, testset, client_config, cid, device, **kwargs)
        temp = [self.count_by_class[cls] if cls in self.count_by_class.keys() else 1e-12 for cls in range(client_config['num_classes'])]
        self.count_by_class_full = torch.tensor(temp).to(self.device)

    def _estimate_prototype(self):
        """
        Estimates prototype vectors based on class distributions.

        Computes the prototype for each class using training data embeddings, 
        normalizes the vectors, and adjusts them for federated aggregation.

        Returns:
            dict: Contains 'scaled_prototype' and 'count_by_class_full'.
        """
        self.model.eval()
        self.model.return_embedding = True
        embedding_dim = self.model.prototype.shape[1]
        prototype = torch.zeros_like(self.model.prototype)
        with torch.no_grad():
            for i, (x, y) in enumerate(self.trainloader):
                x, y = x.to(self.device), y.to(self.device)
                feature_embedding, _ = self.model.forward(x)
                classes_shown_in_this_batch = torch.unique(y).cpu().numpy()
                for cls in classes_shown_in_this_batch:
                    mask = (y == cls)
                    feature_embedding_in_cls = torch.sum(feature_embedding[mask, :], dim=0)
                    prototype[cls] += feature_embedding_in_cls
        for cls in self.count_by_class.keys():
            prototype[cls] /= self.count_by_class[cls]
            prototype_cls_norm = torch.norm(prototype[cls]).clamp(min=1e-12)
            prototype[cls] = torch.div(prototype[cls], prototype_cls_norm)
            prototype[cls] *= self.count_by_class[cls]

        to_share = {'scaled_prototype': prototype, 'count_by_class_full': self.count_by_class_full}
        return to_share

    def _estimate_prototype_adv(self):
        """
        Estimates prototype vectors based on class distributions and densities.

        This method computes the prototypes for each class by taking into account the class densities, 
        adjusting the weights of embeddings based on density, and normalizes the prototype vectors.

        Returns:
            dict: Contains 'adv_agg_prototype', 'count_by_class_full', and 'density_weight'.
        """
        self.model.eval()
        self.model.return_embedding = True
        embeddings = []
        labels = []
        weights = []
        prototype = torch.zeros_like(self.model.prototype)

        with torch.no_grad():
            for i, (x, y) in enumerate(self.trainloader):
                x, y = x.to(self.device), y.to(self.device)
                feature_embedding, logits = self.model.forward(x)
                prob_ = F.softmax(logits, dim=1)
                prob = torch.gather(prob_, dim=1, index=y.view(-1, 1))
                labels.append(y)
                weights.append(prob)
                embeddings.append(feature_embedding)

        embeddings = torch.cat(embeddings, dim=0)
        labels = torch.cat(labels, dim=0)
        weights = torch.cat(weights, dim=0).view(-1, 1)

        class_densities = self.server.calculate_cluster_density(embeddings.cpu().numpy(), labels.cpu().numpy())
        density_weight = torch.tensor([class_densities.get(cls, 1) for cls in range(self.client_config['num_classes'])]).to(self.device)

        for cls in self.count_by_class.keys():
            mask = (labels == cls)
            weights_in_cls = weights[mask, :] * density_weight[cls]
            feature_embedding_in_cls = embeddings[mask, :]

            if torch.sum(weights_in_cls) > 0:
                prototype[cls] = torch.sum(feature_embedding_in_cls * weights_in_cls, dim=0) / torch.sum(weights_in_cls)
            else:
                prototype[cls] = torch.sum(feature_embedding_in_cls, dim=0)

            prototype_cls_norm = torch.norm(prototype[cls]).clamp(min=1e-12)
            prototype[cls] = torch.div(prototype[cls], prototype_cls_norm)

        to_share = {'adv_agg_prototype': prototype, 'count_by_class_full': self.count_by_class_full, 'density_weight': density_weight}
        return to_share

    def upload(self):
        """
        Uploads model state dictionary and prototype estimates.

        Returns:
            tuple: Tuple containing new_state_dict and prototype estimation dictionary.
        """
        if self.client_config['FedNH_client_adv_prototype_agg']:
            return self.new_state_dict, self._estimate_prototype_adv()
        else:
            return self.new_state_dict, self._estimate_prototype()

    def get_embeddings_and_labels(self):
        """
        Retrieves embeddings and labels of training data.

        Returns:
            tuple: Embeddings and labels as torch.Tensors.
        """
        self.model.eval()
        self.model.return_embedding = True
        embeddings = []
        labels = []
        with torch.no_grad():
            for i, (x, y) in enumerate(self.trainloader):
                x, y = x.to(self.device), y.to(self.device)
                feature_embedding, _ = self.model.forward(x)
                embeddings.append(feature_embedding)
                labels.append(y)
        embeddings = torch.cat(embeddings, dim=0)
        labels = torch.cat(labels, dim=0)
        return embeddings, labels


class FedNHServer(FedUHServer):
    """
    Server class for the Federated NH (FedNH) algorithm.

    This class manages client aggregation and serves as a central component in federated learning, 
    handling prototype adjustments and density-based weighting for each round.
    """

    def __init__(self, server_config, clients_dict, exclude, **kwargs):
        """
        Initializes a FedNH server.

        Args:
            server_config (dict): Configuration dictionary for server parameters.
            clients_dict (dict): Dictionary of clients in federated learning.
            exclude (list): List of keys to exclude from aggregation.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(server_config, clients_dict, exclude, **kwargs)
        self.visualization_dir = self.create_simulation_directory()
        self.visualization_rounds = [1, 50, 150, 200]

    def create_simulation_directory(self):
        """
        Creates a new simulation directory under the 'visualization' folder.

        Returns:
            str: Path to the new simulation directory.
        """
        base_dir = os.path.join(os.getcwd(), 'visualization')
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        existing_sim_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        simulation_numbers = [int(d[4:]) for d in existing_sim_dirs if d.startswith('simu') and d[4:].isdigit()]
        next_simulation_number = max(simulation_numbers, default=0) + 1

        new_simulation_dir = os.path.join(base_dir, f'simu{next_simulation_number}')
        os.makedirs(new_simulation_dir)

        print(f"Simulation directory created: {new_simulation_dir}")
        return new_simulation_dir

    def silhouette_analysis(self, valid_embeddings, valid_labels):
        """
        Calculates silhouette score to assess clustering quality.

        Args:
            valid_embeddings (torch.Tensor): Valid embeddings of data points.
            valid_labels (torch.Tensor): Corresponding labels for the embeddings.

        Returns:
            float: Silhouette score or None if insufficient labels.
        """
        unique_labels = torch.unique(valid_labels.cpu())

        if len(unique_labels) < 2:
            return None

        return silhouette_score(valid_embeddings.cpu(), valid_labels.cpu(), metric="cosine")



    def intra_inter_cluster_distances(self, embeddings, labels):
        """
        Calculate the mean intra-cluster and inter-cluster distances for data embeddings.

        Args:
            embeddings (np.ndarray): Array of embedding vectors for data points.
            labels (np.ndarray): Array of class labels for each embedding vector.

        Returns:
            tuple: Average intra-cluster distance and inter-cluster distance. 
                Returns (0, 0) if no valid clusters are found.
        """
        # Initializations for unique label identification and distance arrays
        unique_labels = np.unique(labels)
        intra_distances = []
        inter_distances = []

        # Calculate intra-cluster distances for each class
        for label in unique_labels:
            mask = labels == label
            if np.sum(mask) > 1:  # Requires more than one point for meaningful intra-distance
                points = embeddings[mask]
                intra_distances.append(np.mean(pairwise_distances(points)))

        # Calculate inter-cluster distances between each pair of classes
        if len(unique_labels) > 1:
            for i in range(len(unique_labels)):
                for j in range(i + 1, len(unique_labels)):
                    points_i = embeddings[labels == unique_labels[i]]
                    points_j = embeddings[labels == unique_labels[j]]
                    inter_distances.append(np.mean(pairwise_distances(points_i, points_j)))

        # Return mean intra-cluster and inter-cluster distances
        return np.mean(intra_distances) if intra_distances else 0, np.mean(inter_distances) if inter_distances else 0



    def tsne_visualization(self, prototypes, labels, round_num, tag):
        """
        Visualize class prototypes using t-SNE and save the visualization with convex hulls.

        Args:
            prototypes (torch.Tensor): Prototype vectors representing each class.
            labels (list): List of class labels associated with prototypes.
            round_num (int): Round number to annotate the visualization.
            tag (str): Tag used to differentiate file names for the visualizations (e.g., 'initial', 'final').
        """
        # Setting perplexity based on the number of samples
        num_samples = prototypes.shape[0]
        perplexity = min(30, num_samples - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        tsne_results = tsne.fit_transform(prototypes.cpu().numpy())

        # Plotting with color-coded points and alpha transparency
        plt.figure(figsize=(10, 8))
        for label in set(labels):
            mask = np.array(labels) == label
            plt.scatter(tsne_results[mask, 0], tsne_results[mask, 1], alpha=0.5, label=f'Class {label}')
        plt.title('t-SNE of Class Prototypes')
        plt.legend()
        visualization_path = os.path.join(self.visualization_dir, f"tsne_round_{round_num}_{tag}.png")
        plt.savefig(visualization_path)
        plt.close()
        print(f"t-SNE with alpha transparency saved to {visualization_path}")

        # Plot with convex hulls around clusters for better boundary visualization
        plt.figure(figsize=(10, 8))
        for label in set(labels):
            mask = np.array(labels) == label
            points = tsne_results[mask]
            if len(points) >= 3:
                hull = ConvexHull(points)
                for simplex in hull.simplices:
                    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
            plt.scatter(points[:, 0], points[:, 1], alpha=0.5, label=f'Class {label}')
        plt.title('t-SNE of Class Prototypes with Convex Hulls')
        plt.legend()
        visualization_path = os.path.join(self.visualization_dir, f"tsne_round_{round_num}_{tag}_hulls.png")
        plt.savefig(visualization_path)
        plt.close()
        print(f"t-SNE with convex hulls saved to {visualization_path}")

    def exclude_outliers(data, proportion=0.10):
        """
        Exclude a proportion of the farthest data points based on distance from the centroid.

        Args:
            data (np.ndarray): Array of data points to filter.
            proportion (float): Proportion of data points to exclude, default is 10%.

        Returns:
            np.ndarray: Filtered data with outliers removed.
        """
        from scipy.spatial.distance import cdist
        centroid = np.mean(data, axis=0)
        distances = cdist(data, [centroid], 'euclidean')
        sorted_distances = np.argsort(distances.flatten())
        threshold = int(len(data) * (1 - proportion))
        return data[sorted_distances[:threshold]]

    def calculate_silhouette_score(data, labels):
        """
        Computes the silhouette score, indicating how well each sample is assigned to its cluster.
        
        Args:
            data (np.ndarray): The data points to analyze.
            labels (np.ndarray): Cluster labels for each data point.
        
        Returns:
            float: The silhouette score of the clustering.
        """
        score = silhouette_score(data, labels)
        return score

    def client_distribution_visualization(self, client_uploads, round_num):
        """
        Visualizes the distribution of classes across clients.
        
        Args:
            client_uploads (list): List of client uploads containing prototype estimates.
            round_num (int): Current round number for visualization.
        """
        num_classes = self.server_config['num_classes']
        client_counts = [upload[1]['count_by_class_full'].cpu().numpy() for upload in client_uploads if 'count_by_class_full' in upload[1]]
        
        plt.figure(figsize=(10, 8))
        for i, client_count in enumerate(client_counts):
            plt.bar(range(num_classes), client_count, alpha=0.5, label=f'Class {i}')
        
        plt.xlabel('Client')
        plt.ylabel('Number of Samples')
        plt.title('Classes Distribution Across Clients')
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        visualization_path = os.path.join(self.visualization_dir, f"client_distribution_round_{round_num}.png")

        try:
            plt.savefig(visualization_path)
            plt.close()
            print(f"Client distribution visualization saved to {visualization_path}")

            if os.path.exists(visualization_path):
                print(f"Verified that the file {visualization_path} exists.")
            else:
                print(f"Error: The file {visualization_path} does not exist after saving.")
        except Exception as e:
            print(f"Failed to save client distribution visualization: {e}")


    def prototype_weight_visualization(self, prototype_weights, round_num, tag=''):
        """
        Visualizes prototype weights for each class.
        
        Args:
            prototype_weights (torch.Tensor): Prototype weights per class.
            round_num (int): Current round number for visualization.
            tag (str): Tag to specify file details (e.g., 'pre_norm', 'post_norm').
        """
        num_classes = prototype_weights.shape[0]
        
        plt.figure(figsize=(10, 8))
        plt.bar(range(num_classes), prototype_weights.cpu().numpy(), alpha=0.7)
        
        plt.xlabel('Class')
        plt.ylabel('Prototype Weight')
        plt.title(f'Prototype Weights for Each Class ({tag})')
        visualization_path = os.path.join(self.visualization_dir, f"prototype_weights_round_{round_num}_{tag}.png")

        try:
            plt.savefig(visualization_path)
            plt.close()
            print(f"Prototype weights visualization saved to {visualization_path}")

            if os.path.exists(visualization_path):
                print(f"Verified that the file {visualization_path} exists.")
            else:
                print(f"Error: The file {visualization_path} does not exist after saving.")
        except Exception as e:
            print(f"Failed to save prototype weights visualization: {e}")


    def calculate_cluster_density(self, embeddings, labels):
        """
        Calculates normalized cluster density per class after removing 10% of the most distant points.
        
        Args:
            embeddings (np.ndarray): Data points embeddings.
            labels (np.ndarray): Class labels of data points.
        
        Returns:
            dict: Class density normalized values.
        """
        class_densities = {}
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            mask = labels == label
            if np.sum(mask) > 1:
                points = embeddings[mask]
                nbrs = NearestNeighbors(n_neighbors=2).fit(points)
                distances, _ = nbrs.kneighbors(points)

                sorted_distances = np.sort(distances[:, 1])
                cutoff_index = int(0.9 * len(sorted_distances))
                trimmed_distances = sorted_distances[:cutoff_index]

                mean_distance = np.mean(np.log1p(trimmed_distances))
                density = np.exp(-mean_distance)
                class_densities[label] = density
            else:
                class_densities[label] = 1e-6

        all_densities = np.array(list(class_densities.values())).reshape(-1, 1)
        scaler = QuantileTransformer(output_distribution='uniform')
        normalized_densities = scaler.fit_transform(all_densities).flatten()

        for i, label in enumerate(unique_labels):
            class_densities[label] = normalized_densities[i]

        return class_densities

    def visualize_cluster_density(self, embeddings, labels, round_num):
        """
        Visualize cluster density using a bar plot.

        Args:
        - embeddings (torch.Tensor): Embeddings of the data points.
        - labels (torch.Tensor): Labels corresponding to the embeddings.
        - round_num (int): Current round number for visualization.
        """
        embeddings = embeddings.cpu().numpy()
        labels = labels.cpu().numpy()
        class_densities = self.calculate_cluster_density(embeddings, labels)

        plt.figure(figsize=(10, 8))
        plt.bar(class_densities.keys(), class_densities.values(), alpha=0.7)
        plt.xlabel('Class')
        plt.ylabel('Cluster Density')
        plt.title(f'Cluster Density for Each Class (Round {round_num})')
        visualization_path = os.path.join(self.visualization_dir, f"cluster_density_round_{round_num}.png")

        try:
            plt.savefig(visualization_path)
            plt.close()
            print(f"Cluster density visualization saved to {visualization_path}")

            # Verify the file has been created
            if os.path.exists(visualization_path):
                print(f"Verified that the file {visualization_path} exists.")
            else:
                print(f"Error: The file {visualization_path} does not exist after saving.")
        except Exception as e:
            print(f"Failed to save cluster density visualization: {e}")

    def visualize_client_data_clusters(self, client_uploads, round_num, num_points_per_class=100):
        """
        Visualizes client data clusters using t-SNE, excluding class prototypes.

        Args:
        - client_uploads (list): List of client uploads.
        - round_num (int): Current round number for viewing.
        - num_points_per_class (int): Number of points to sample per class.
        """
        all_embeddings = []
        all_labels = []
        for client in self.clients_dict.values():
            embeddings, labels = client.get_embeddings_and_labels()
            all_embeddings.append(embeddings)
            all_labels.append(labels)

        all_embeddings = torch.cat(all_embeddings, dim=0).cpu().numpy()
        all_labels = torch.cat(all_labels, dim=0).cpu().numpy()

        # Sample points for better readability
        sampled_embeddings = []
        sampled_labels = []
        unique_labels = np.unique(all_labels)
        for label in unique_labels:
            label_indices = np.where(all_labels == label)[0]
            sampled_indices = np.random.choice(label_indices, min(num_points_per_class, len(label_indices)), replace=False)
            sampled_embeddings.append(all_embeddings[sampled_indices])
            sampled_labels.append(all_labels[sampled_indices])

        sampled_embeddings = np.vstack(sampled_embeddings)
        sampled_labels = np.hstack(sampled_labels)

        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        tsne_results = tsne.fit_transform(sampled_embeddings)

        plt.figure(figsize=(10, 8))
        for label in set(sampled_labels):
            mask = sampled_labels == label
            plt.scatter(tsne_results[mask, 0], tsne_results[mask, 1], alpha=0.5, label=f'Class {label}')
        plt.title('t-SNE of Client Data Clusters')
        plt.legend()
        visualization_path = os.path.join(self.visualization_dir, f"client_data_clusters_round_{round_num}.png")
        plt.savefig(visualization_path)
        plt.close()
        print(f"Client data clusters t-SNE saved to {visualization_path}")

        # Interactive plot using Plotly
        fig = px.scatter(x=tsne_results[:, 0], y=tsne_results[:, 1], color=sampled_labels, title='t-SNE of Client Data Clusters')
        visualization_path = os.path.join(self.visualization_dir, f"client_data_clusters_round_{round_num}_interactive.html")
        fig.write_html(visualization_path)
        print(f"Interactive t-SNE of client data clusters saved to {visualization_path}")

        # Visualize cluster density
        self.visualize_cluster_density(torch.tensor(sampled_embeddings), torch.tensor(sampled_labels), round_num)

    
    def visualize_client_clusters_separately(self, client_uploads, round_num, num_points_per_class=100):
        """
        Visualizes client data clusters separately for each client.

        Args:
        - client_uploads (list): List of client uploads.
        - round_num (int): Current round number for visualization.
        - num_points_per_class (int): Number of points to sample per class.
        """
        for client_idx, client in enumerate(list(self.clients_dict.values())[:10]):
            embeddings, labels = client.get_embeddings_and_labels()
            embeddings = embeddings.cpu().numpy()
            labels = labels.cpu().numpy()

            # Sample points for better readability
            sampled_embeddings = []
            sampled_labels = []
            unique_labels = np.unique(labels)
            for label in unique_labels:
                label_indices = np.where(labels == label)[0]
                sampled_indices = np.random.choice(label_indices, min(num_points_per_class, len(label_indices)), replace=False)
                sampled_embeddings.append(embeddings[sampled_indices])
                sampled_labels.append(labels[sampled_indices])

            sampled_embeddings = np.vstack(sampled_embeddings)
            sampled_labels = np.hstack(sampled_labels)

            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            tsne_results = tsne.fit_transform(sampled_embeddings)

            plt.figure(figsize=(10, 8))
            for label in set(sampled_labels):
                mask = sampled_labels == label
                plt.scatter(tsne_results[mask, 0], tsne_results[mask, 1], alpha=0.5, label=f'Class {label}')
            plt.title(f't-SNE of Client {client_idx + 1} Data Clusters (Round {round_num})')
            plt.legend()
            visualization_path = os.path.join(self.visualization_dir, f"client_{client_idx + 1}_data_clusters_round_{round_num}.png")
            plt.savefig(visualization_path)
            plt.close()
            print(f"Client {client_idx + 1} data clusters t-SNE saved to {visualization_path}")

            # Interactive plot using Plotly
            fig = px.scatter(x=tsne_results[:, 0], y=tsne_results[:, 1], color=sampled_labels,
                             title=f't-SNE of Client {client_idx + 1} Data Clusters (Round {round_num})')
            visualization_path = os.path.join(self.visualization_dir, f"client_{client_idx + 1}_data_clusters_round_{round_num}_interactive.html")
            fig.write_html(visualization_path)
            print(f"Interactive t-SNE of client {client_idx + 1} data clusters saved to {visualization_path}")

            # Visualize cluster density
            self.visualize_cluster_density(torch.tensor(sampled_embeddings), torch.tensor(sampled_labels), round_num)
     
     
    def save_metrics_to_file(self, metrics):
        """
        Saves metrics to a text file.

        Args:
        - metrics (dict): Dictionary containing metrics for the round.
        """
        metrics_path = os.path.join(self.visualization_dir, f"metrics_round_{metrics['round']}.txt")
        with open(metrics_path, 'w') as f:
            f.write(f'Silhouette Score: {metrics["silhouette"]}\n')
            f.write(f'Intra-cluster Distance: {metrics["intra_distance"]}\n')
            f.write(f'Inter-cluster Distance: {metrics["inter_distance"]}\n')
        print(f"Metrics saved to {metrics_path}")

    def visualize_metrics(self):
        """
        Visualizes metrics (intra-cluster and inter-cluster distances) over rounds.

        """
        rounds = [m['round'] for m in self.metrics_history]
        intra_distances = [m['intra_distance'] for m in self.metrics_history]
        inter_distances = [m['inter_distance'] for m in self.metrics_history]

        width = 0.2
        x = np.arange(len(rounds))

        plt.figure(figsize=(10, 6))
        plt.bar(x, intra_distances, width, label='Intra-cluster Distance', color='g')
        plt.bar(x + width, inter_distances, width, label='Inter-cluster Distance', color='r')

        plt.xlabel('Round')
        plt.ylabel('Scores')
        plt.title('Clustering Metrics over Rounds')
        plt.xticks(x, rounds)
        plt.legend()
        plt.tight_layout()

        metrics_visualization_path = os.path.join(self.visualization_dir, 'metrics_visualization.png')
        plt.savefig(metrics_visualization_path)
        plt.close()
        print(f"Metrics visualization saved to {metrics_visualization_path}")
            
        
    def visualize_client_cluster_density(self, round_num, num_points_per_class=100):
        """
        Visualizes cluster density for each client separately using a bar plot.

        Args:
        - round_num (int): Current round number for visualization.
        - num_points_per_class (int): Number of points to sample per class.
        """
        for client_idx, client in enumerate(list(self.clients_dict.values())[:10]):
            embeddings, labels = client.get_embeddings_and_labels()
            embeddings = embeddings.cpu().numpy()
            labels = labels.cpu().numpy()

            # Sample points for better readability
            sampled_embeddings = []
            sampled_labels = []
            unique_labels = np.unique(labels)
            for label in unique_labels:
                label_indices = np.where(labels == label)[0]
                sampled_indices = np.random.choice(label_indices, min(num_points_per_class, len(label_indices)), replace=False)
                sampled_embeddings.append(embeddings[sampled_indices])
                sampled_labels.append(labels[sampled_indices])

            sampled_embeddings = np.vstack(sampled_embeddings)
            sampled_labels = np.hstack(sampled_labels)

            class_densities = self.calculate_cluster_density(sampled_embeddings, sampled_labels)

            plt.figure(figsize=(10, 8))
            plt.bar(class_densities.keys(), class_densities.values(), alpha=0.7)
            plt.xlabel('Class')
            plt.ylabel('Cluster Density')
            plt.title(f'Cluster Density for Client {client_idx + 1} (Round {round_num})')
            visualization_path = os.path.join(self.visualization_dir, f"client_{client_idx + 1}_cluster_density_round_{round_num}.png")

            try:
                plt.savefig(visualization_path)
                plt.close()
                print(f"Client {client_idx + 1} cluster density visualization saved to {visualization_path}")

                # Verify the file has been created
                if os.path.exists(visualization_path):
                    print(f"Verified that the file {visualization_path} exists.")
                else:
                    print(f"Error: The file {visualization_path} does not exist after saving.")
            except Exception as e:
                print(f"Failed to save cluster density visualization: {e}")
            
            
    # def confusion_matrix_visualization(self, embeddings, labels, round_num):
    #     """
    #     Visualizes the confusion matrix based on the model predictions.

    #     Args:
    #     - embeddings (torch.Tensor): Embeddings of the data points.
    #     - labels (torch.Tensor): True labels corresponding to the embeddings.
    #     - round_num (int): Current round number for visualization.
    #     """
    #     # Get model predictions
    #     self.model.eval()
    #     self.model.return_embedding = False
    #     all_preds = []
    #     all_labels = []
    #     with torch.no_grad():
    #         for i, (x, y) in enumerate(self.trainloader):
    #             x = x.to(self.device)
    #             outputs = self.model(x)
    #             _, preds = torch.max(outputs, 1)
    #             all_preds.append(preds.cpu().numpy())
    #             all_labels.append(y.cpu().numpy())

    #     all_preds = np.concatenate(all_preds)
    #     all_labels = np.concatenate(all_labels)
    #     cm = confusion_matrix(all_labels, all_preds, labels=range(self.server_config['num_classes']))

    #     plt.figure(figsize=(10, 8))
    #     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
    #                 xticklabels=list(range(self.server_config['num_classes'])),
    #                 yticklabels=list(range(self.server_config['num_classes'])))
    #     plt.xlabel('Predicted Label')
    #     plt.ylabel('True Label')
    #     plt.title(f'Confusion Matrix (Round {round_num})')
    #     visualization_path = os.path.join(self.visualization_dir, f"confusion_matrix_round_{round_num}.png")
    #     plt.savefig(visualization_path)
    #     plt.close()
    #     print(f"Confusion matrix visualization saved to {visualization_path}")
        
        
    def aggregate(self, client_uploads, round):
        """
        Aggregates updates from clients to update the server model, taking into account the class densities.

        Args:
        - client_uploads (list): A list of tuples containing the state dictionaries and prototype dictionaries from clients.
        - round (int): The current round number of the federated learning process.
        """
        server_lr = self.server_config['learning_rate'] * (self.server_config['lr_decay_per_round'] ** (round - 1))
        num_participants = len(client_uploads)

        update_direction_state_dict = None
        cumsum_per_class = torch.zeros(self.server_config['num_classes']).to(self.clients_dict[0].device)
        agg_weights_vec_dict = {}

        initial_prototypes = self.server_model_state_dict['prototype'].clone()
        self.tsne_visualization(initial_prototypes, list(range(self.server_config['num_classes'])), round, tag='initial')

        with torch.no_grad():
            for idx, (client_state_dict, prototype_dict) in enumerate(client_uploads):
                if self.server_config['FedNH_server_adv_prototype_agg']:
                    if 'adv_agg_prototype' in prototype_dict:
                        mu = prototype_dict['adv_agg_prototype']
                        W = self.server_model_state_dict['prototype']
                        density_weight = prototype_dict['density_weight']  # Use density weights
                        agg_weights_vec_dict[idx] = density_weight * torch.exp(torch.sum(W * mu, dim=1, keepdim=True))

                    else:
                        raise KeyError(f"Client {idx} did not provide 'adv_agg_prototype'. Check client configuration.")
                else:
                    if 'scaled_prototype' in prototype_dict:
                        cumsum_per_class += prototype_dict['count_by_class_full']
                    else:
                        raise KeyError(f"Client {idx} did not provide 'scaled_prototype'. Check client configuration.")
                
                client_update = linear_combination_state_dict(client_state_dict,
                                                            self.server_model_state_dict,
                                                            1.0,
                                                            -1.0,
                                                            exclude=self.exclude_layer_keys)
                if idx == 0:
                    update_direction_state_dict = client_update
                else:
                    update_direction_state_dict = linear_combination_state_dict(update_direction_state_dict,
                                                                                client_update,
                                                                                1.0,
                                                                                1.0,
                                                                                exclude=self.exclude_layer_keys)

            self.server_model_state_dict = linear_combination_state_dict(self.server_model_state_dict,
                                                                        update_direction_state_dict,
                                                                        1.0,
                                                                        server_lr / num_participants,
                                                                        exclude=self.exclude_layer_keys)

            avg_prototype = torch.zeros_like(self.server_model_state_dict['prototype'])
            if self.server_config['FedNH_server_adv_prototype_agg']:
                m = self.server_model_state_dict['prototype'].shape[0]
                sum_of_weights = torch.zeros((m, 1)).to(avg_prototype.device)
                for idx, (_, prototype_dict) in enumerate(client_uploads):
                    sum_of_weights += agg_weights_vec_dict[idx]
                    avg_prototype += agg_weights_vec_dict[idx] * prototype_dict['adv_agg_prototype']
                avg_prototype /= sum_of_weights
            else:
                for _, prototype_dict in client_uploads:
                    avg_prototype += prototype_dict['scaled_prototype'] / cumsum_per_class.view(-1, 1)

            pre_norm_weights = avg_prototype.norm(dim=1)
            self.prototype_weight_visualization(pre_norm_weights, round, tag='pre_norm')

            avg_prototype = F.normalize(avg_prototype, dim=1)
            weight = self.server_config['FedNH_smoothing']
            temp = weight * self.server_model_state_dict['prototype'] + (1 - weight) * avg_prototype
            self.server_model_state_dict['prototype'].copy_(F.normalize(temp, dim=1))

            post_norm_weights = self.server_model_state_dict['prototype'].norm(dim=1)
            self.prototype_weight_visualization(post_norm_weights, round, tag='post_norm')

            self.tsne_visualization(temp, list(range(self.server_config['num_classes'])), round, tag='final')
            self.visualize_client_clusters_separately(client_uploads, round)
            self.visualize_client_cluster_density(round)
            self.client_distribution_visualization(client_uploads, round)
            self.visualize_client_data_clusters(client_uploads, round)
            # # Visualize client distribution and data clusters
            # if round in self.visualization_rounds:
            #     self.client_distribution_visualization(client_uploads, round)
            #     self.visualize_client_data_clusters(client_uploads, round)