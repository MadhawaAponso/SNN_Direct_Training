import numpy as np
import os
import torch
from torch.utils.data.dataloader import default_collate
from torch.utils.data import SequentialSampler
import torch.utils.data.distributed

# SpikingjellyDataset Class
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

# Loader Class for Batch Processing
class Loader:
    def __init__(self, dataset, args, device, distributed, batch_size, drop_last=False):
        self.device = device
        if distributed is True:
            self.sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            self.sampler = torch.utils.data.RandomSampler(dataset)
        
        self.loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=self.sampler,
                                                  num_workers=args.train_num_workers, pin_memory=True,
                                                  collate_fn=collate_events, drop_last=drop_last)

    def __iter__(self):
        for data in self.loader:
            data = [d.to(self.device) for d in data]
            yield data

    def __len__(self):
        return len(self.loader)

# Collate function to handle batching of events
def collate_events(data):
    labels = []
    events = []
    folder_names = []
    for i, d in enumerate(data):
        labels.append(d[1])
        folder_names.append(d[2])
        ev = torch.cat([d[0], i * torch.ones((len(d[0]), 1), dtype=torch.float32)], 1)
        events.append(ev)
    events = torch.cat(events, 0)
    labels = default_collate(labels)
    folder_names = default_collate(folder_names)
    return events, labels, folder_names

# Define arguments (replace with actual values)
class Args:
    train_num_workers = 4  # Adjust based on system performance

args = Args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
distributed = False  # Set to True if using Distributed Training
batch_size = 16  # Adjust based on available memory

# Main entry point
if __name__ == "__main__":
    # Load dataset
    dataset_path = "C:/Users/Madhawa/Desktop/backupNPZ/npz_events"
    sj = SpikingjellyDataset(dataset_path)

    # Initialize data loader
    data_loader = Loader(dataset=sj, args=args, device=device, distributed=distributed, batch_size=batch_size)

    # Check dataset size
    print("Dataset size:", len(sj))

    # Print first sample from the first batch
    for batch in data_loader:
        events, labels, folder_names = batch
        print("Events shape:", events.shape)
        print("Labels:", labels)
        print("Folder Names:", folder_names)
        break  # Print only the first batch
