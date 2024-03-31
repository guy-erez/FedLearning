import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor
from NnModel import NeuralNetwork,get_device
import NnModel 
from dataset import DemoDataset, ToTensor

# Set random seed for reproducibility
torch.manual_seed(27)
dataset = DemoDataset(csv_file="Anonymous_Demo_Data.csv", label='death', drop = ['reference_date_internal', 'site'],transform=ToTensor())
training_data, test_data = random_split(dataset, [0.8, 0.2])

client1_size = int(0.00 * len(training_data))
client2_size = len(training_data) - client1_size

client1_dataset, client2_dataset = random_split(training_data, [client1_size, client2_size])
datasets_list = [client1_dataset, client2_dataset]

batch_size = 64
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for i,dataset in enumerate(datasets_list):        
    # Create data loaders.
    train_dataloader = DataLoader(dataset, batch_size=batch_size)
    
    # get avilable device
    device = get_device()
    print(f"Using {device} device")
    
    # define model, loss and optimizer
    model = NeuralNetwork().to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # train loop 
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        NnModel.train(train_dataloader, model, loss_fn, optimizer, device)
        NnModel.test(test_dataloader, model, loss_fn, device)
    print(f"final acc for model {i}")
    NnModel.test(test_dataloader, model, loss_fn, device)
    model_path = f"model_{i}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved PyTorch Model State to {model_path}")