import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
from NnModel import NeuralNetwork,get_device
import NnModel
from dataset import DemoDataset, ToTensor

# Set random seed for reproducibility
torch.manual_seed(27)
dataset = DemoDataset(csv_file="Anonymous_Demo_Data.csv", label='death', drop = ['reference_date_internal', 'site'],transform=ToTensor())
training_data, test_data = random_split(dataset, [0.8, 0.2])
test_dataloader = DataLoader(test_data, batch_size=64)


device = get_device()
model_0 = NeuralNetwork().to(device)
model_1 = NeuralNetwork().to(device)
model_list = [model_0,model_1]
common_model = NeuralNetwork().to(device)

model_0.load_state_dict(torch.load("model_0.pth"))
model_1.load_state_dict(torch.load("model_1.pth"))

with torch.no_grad():
    for name, param in common_model.named_parameters():
        param.data = torch.zeros_like(param)
    for model in model_list:
        print(common_model.linear_relu_stack[0].weight[0])
        for common_param, param in zip(common_model.parameters(),model.parameters()):
            common_param.data += param
    print(common_model.linear_relu_stack[0].weight[0])
    for name, param in common_model.named_parameters():
        param.data /= len(model_list)
  
print(common_model.linear_relu_stack[0].weight[0])

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