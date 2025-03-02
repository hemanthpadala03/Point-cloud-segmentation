import numpy as np
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch import nn
import pptk

# Function to load and preprocess point cloud data
def load_and_preprocess_data(file_path):
    # Read the .bin file
    pc_bin = np.fromfile(file_path, '<f4')
    pc_bin = np.reshape(pc_bin, (-1, 4))

    # Filter based on Z-axis range
    pc_bin = pc_bin[(pc_bin[:, 2] >= -1.5) & (pc_bin[:, 2] <= 1.5)]

    # Further filter based on X and Y axis range
    pc_bin = pc_bin[(pc_bin[:, 0] >= -10) & (pc_bin[:, 0] <= 10)]
    pc_bin = pc_bin[(pc_bin[:, 1] >= -10) & (pc_bin[:, 1] <= 10)]

    # Normalize features (X, Y, Z coordinates)
    features = pc_bin[:, :3]
    features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
    print(features.shape)  # Print shape of features

    return features, pc_bin

# Function to calculate the Euclidean distance between points
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Function to create the KNN graph
def create_knn_graph_custom(features, k=10):
    num_points = features.shape[0]
    adj_matrix = np.zeros((num_points, num_points))

    # Calculate distances and find k-nearest neighbors
    for i in range(num_points):
        distances = np.array([euclidean_distance(features[i], features[j]) for j in range(num_points)])
        knn_indices = np.argsort(distances)[1:k+1]  # Exclude the point itself (distance 0)
        adj_matrix[i, knn_indices] = 1
        adj_matrix[knn_indices, i] = 1  # Ensure the graph is undirected

    edge_index = np.array(np.nonzero(adj_matrix))
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    return edge_index

# Load and preprocess point cloud data
file_path = './Kitti_dataset/training/velodyne/000000.bin'
features, pc_bin = load_and_preprocess_data(file_path)

# Create edge index for k-nearest neighbor graph
edge_index_custom = create_knn_graph_custom(features, k=10)
print(edge_index_custom.shape)  # Print shape of edge index

# Convert features to torch tensor
x = torch.tensor(features, dtype=torch.float)

# Create PyTorch Geometric data object
data_custom = Data(x=x, edge_index=edge_index_custom)

# Define GNN model class with a more complex architecture
class GNNpool(nn.Module):
    def __init__(self, input_dim, conv_hidden, mlp_hidden, num_clusters, device):
        super(GNNpool, self).__init__()
        self.device = device
        self.num_clusters = num_clusters
        self.mlp_hidden = mlp_hidden

        # GNN conv layers
        self.conv1 = pyg_nn.GCNConv(input_dim, conv_hidden)
        self.conv2 = pyg_nn.GCNConv(conv_hidden, conv_hidden)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(conv_hidden, mlp_hidden), 
            nn.ELU(), 
            nn.Dropout(0.5),  # Increase dropout rate
            nn.Linear(mlp_hidden, self.num_clusters)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)  # Applying first convolution
        x = F.elu(x)
        x = self.conv2(x, edge_index)  # Applying second convolution
        x = F.elu(x)

        # Pass features through MLP
        H = self.mlp(x)
        # Cluster assignment for matrix S
        S = F.softmax(H, dim=1)

        return S

    def loss(self, A, S):
        # Cut loss
        A_pool = torch.matmul(torch.matmul(A, S).t(), S)
        num = torch.trace(A_pool)

        D = torch.diag(torch.sum(A, dim=-1))
        D_pooled = torch.matmul(torch.matmul(D, S).t(), S)
        den = torch.trace(D_pooled)
        mincut_loss = -(num / den)

        # Orthogonality loss
        St_S = torch.matmul(S.t(), S)
        I_S = torch.eye(self.num_clusters, device=self.device)
        ortho_loss = torch.norm(St_S / torch.norm(St_S) - I_S / torch.norm(I_S))

        return mincut_loss + ortho_loss

# Training parameters
input_dim = 3  # (x, y, z)
conv_hidden = 64  # Increase hidden units in GCNConv layer
mlp_hidden = 128  # Increase hidden units in MLP
num_clusters = 15  # Number of clusters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available

# Instantiate model
model = GNNpool(input_dim, conv_hidden, mlp_hidden, num_clusters, device).to(device)

# Move data to device
data_custom = data_custom.to(device)

# Training loop
epochs = 50  # Increase the number of epochs
print_every = 10  # Print loss every 10 epochs

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)  # Decrease learning rate
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)  # Learning rate scheduler

for epoch in range(epochs):
    # Forward pass
    S = model(data_custom)
    
    # Example adjacency matrix (identity for simplicity)
    A = torch.eye(features.shape[0]).to(device)
    
    # Calculate loss
    loss = model.loss(A, S)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    # Print loss every print_every epochs
    if (epoch + 1) % print_every == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')

print('Training finished.')

# Get the cluster assignments
_, cluster_assignments = torch.max(S, dim=1)
cluster_assignments = cluster_assignments.cpu().numpy()

# Visualize segmented data using pptk
def visualize_clusters(pc_bin, cluster_assignments):
    num_clusters = np.max(cluster_assignments) + 1
    for i in range(num_clusters):
        mask = cluster_assignments == i
        viewer = pptk.viewer(pc_bin[mask, :3])
        viewer.set(point_size=0.01)
        print(i)
        input("Press Enter to view the next cluster...")

visualize_clusters(pc_bin, cluster_assignments)
