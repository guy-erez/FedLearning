import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
from NnModel import NeuralNetwork,get_device
import NnModel
from dataset import DemoDataset, ToTensor
import argparse 
import os

def main(args):
    dataset = DemoDataset(csv_file=args.dataset, label='death', drop = ['reference_date_internal', 'site'],transform=ToTensor())
    training_data, test_data = random_split(dataset, [0.8, 0.2])
    test_dataloader = DataLoader(test_data, batch_size=64)

    device = get_device()
    client_models_pth = os.listdir(args.clientModels)
    model_list = []

    for i,client_pth in enumerate(client_models_pth):
        client_pth_path = os.path.join(args.clientModels,client_pth)
        model = NeuralNetwork().to(device)
        model.load_state_dict(torch.load(client_pth_path))
        model_list.append(model)
    
    common_model = NeuralNetwork().to(device)

    with torch.no_grad():
        # init common model with zeros 
        for name, param in common_model.named_parameters():
            param.data = torch.zeros_like(param)
        # sum client models    
        for model in model_list:
            # print(common_model.linear_relu_stack[0].weight[0])
            for common_param, param in zip(common_model.parameters(),model.parameters()):
                common_param.data += param
        # print(common_model.linear_relu_stack[0].weight[0])
        # divide each weight  
        for name, param in common_model.named_parameters():
            param.data /= len(model_list)
    
    # print(common_model.linear_relu_stack[0].weight[0])

    classes = ['Alive', 'Dead'] 

    common_model.eval()
    sample = test_data[0]
    with torch.no_grad():
        x = sample['x'].to(device,dtype=torch.float32)
        pred = common_model(x)
        predicted, actual = classes[pred.to(int)], classes[sample['y'].to(int)]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')
        loss_fn = nn.BCEWithLogitsLoss()
        print("unified model:")
        NnModel.test(test_dataloader, common_model, loss_fn, device)
        for i, model in enumerate(model_list):
            print(f"model {i}:")
            NnModel.test(test_dataloader, model, loss_fn, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A simple Python script with a main function and a user argument.')
    parser.add_argument('--dataset', type=str, help='path to dataset')
    parser.add_argument('--clientModels', type=str, help='path to client models dir')

    args = parser.parse_args()
    main(args)