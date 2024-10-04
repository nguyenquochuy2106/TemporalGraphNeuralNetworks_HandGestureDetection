# model_training.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.datasets import Planetoid

class TGNNModel(nn.Module):
    def __init__(self, num_features, num_classes):
        super(TGNNModel, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, 16)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x

def train_model(dataset, model_save_path='saved_models'):
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prepare DataLoader
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model
    model = TGNNModel(num_features=dataset.num_node_features, num_classes=dataset.num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(100):
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data.x, data.edge_index, data.batch)
            loss = criterion(output, data.y)
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    # Save the best model
    torch.save(model.state_dict(), os.path.join(model_save_path, 'best_model.pth'))
    print("Model training complete. Best model saved.")

if __name__ == "__main__":
    from torch_geometric.datasets import Planetoid
    dataset = Planetoid(root='data/Cora', name='Cora')  # Load your dataset here
    train_model(dataset)
