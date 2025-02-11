import numpy as np
import torch
import os

class SpikingjellyDataset:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.files = []  # Store (file_path, label) tuples

        # Iterate through each folder (gesture class)
        for class_label, class_folder in enumerate(sorted(os.listdir(dataset_path))):
            class_path = os.path.join(dataset_path, class_folder)
            if os.path.isdir(class_path):  # Ensure it's a directory
                for file in sorted(os.listdir(class_path)):
                    if file.endswith(".npz"):  # Only use .npz files
                        self.files.append((os.path.join(class_path, file), class_label))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path, label = self.files[idx]  # Get file path and label
        data = np.load(file_path, allow_pickle=True)

        # Ensure keys exist
        required_keys = {"x", "y", "t", "p", "f"}
        if not required_keys.issubset(data.files):
            raise ValueError(f"Missing keys in {file_path}: {set(data.files) - required_keys}")

        x = data["x"].astype(np.float32)
        y = data["y"].astype(np.float32)
        t = data["t"].astype(np.float32)
        p = data["p"].astype(np.float32)
        folder_name = data["f"].item()

        events = np.stack([x, y, t, p], axis=1)  # Shape: (num_events, 4)
        
        return torch.from_numpy(events), label, folder_name

# Load dataset
dataset_path = "C:/Users/Madhawa/Desktop/backupNPZ/npz_events"
sj = SpikingjellyDataset(dataset_path)

# Check dataset size
print("Dataset size:", len(sj))

# Print first sample
print(sj[1])  # Prints (events tensor, label, folder_name)

# for i in range(len(sj)):  # Iterate through dataset
#     events, label, folder_name = sj[i]
#     if label == 1:
#         print(f"Found sample at index {i}")
#         print("Events Shape:", events.shape)
#         print("Folder Name:", folder_name)
#         break  # Stop after finding the first match
print(sj[1459])