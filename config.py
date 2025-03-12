# config.py

# Other constants
batch_size = 64
num_clients = 5

# Training parameters
num_epochs = 3

# Directory paths
dataset_path = "./texas100.npz"
splits = f"dataset_splits{num_epochs}.json"
local_model_path = f"local/models{num_epochs}"
global_model_path = f"global/models{num_epochs}"
combined_snapshots_path = f"combined_model_snapshots{num_epochs}.pth"
