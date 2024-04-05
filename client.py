import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor
from NnModel import NeuralNetwork,get_device
import NnModel 
from dataset import DemoDataset, ToTensor
import argparse
import os 

def  main(args):
    dataset = DemoDataset(csv_file=args.dataset, label='death', drop = ['reference_date_internal', 'site'], transform=ToTensor())
    training_data, test_data = random_split(dataset, [0.8, 0.2])
    batch_size = 64
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    i = args.id 
    # Create data loader
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    
    # get avilable device
    device = get_device()
    print(f"Using {device} device")
    
    # define model, loss and optimizer
    model = NeuralNetwork().to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # train loop 
    epochs = args.epochs
    os.makedirs('clientModels', exist_ok=True)
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        NnModel.train(train_dataloader, model, loss_fn, optimizer, device)
        NnModel.test(test_dataloader, model, loss_fn, device)
    print(f"final acc for model {i}")
    NnModel.test(test_dataloader, model, loss_fn, device)
    model_path = f"clientModels/model_{i}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved PyTorch Model State to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A simple Python script with a main function and a user argument.')
    parser.add_argument('--id', type=int, help='client id')
    parser.add_argument('--dataset', type=str, help='path to dataset')
    parser.add_argument('--epochs', type=int, default=5, help='amount of epochs to train the model')

    args = parser.parse_args()
    main(args)


