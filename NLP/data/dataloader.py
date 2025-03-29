from torch.utils.data import DataLoader
from custom_dataset import CustomDataset

def create_data_loader(file_path, tokenizer, batch_size=32, max_len=256):
    dataset = CustomDataset(file_path, tokenizer, max_len)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return data_loader
