import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from matplotlib import pyplot as plt
from VMDataset import VideoSoundDataset
from torch.utils.data import Subset
import argparse
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.resnet import ResNet50_Weights
import numpy as np
import torchaudio

class SiameseConvNet(nn.Module):
    def __init__(self):
        super(SiameseConvNet, self).__init__()
        # Shared convolutional layers
        self.convnet = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2),  # reduces size to 112x112
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),  # reduces size to 56x56
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),  # reduces size to 28x28

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 14 * 14 * 2, 1024),  # *2 because of concatenation
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        
        # Split input image
        x1 = x[:,:,:,:224]
        x2 = x[:,:,:,224:]
        # Each branch processes an image
        out1 = self.convnet(x1)
        out2 = self.convnet(x2)

        # Flatten and concatenate features from both images
        out1 = out1.view(out1.size(0), -1)
        out2 = out2.view(out2.size(0), -1)
        out = torch.cat((out1, out2), dim=1)  # Concatenate along the feature dimension

        # Fully connected layers
        out = self.fc_layers(out)
        
        return out

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()

#         self.conv1 = nn.Conv2d(3,64,3,padding=1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Use a pre-trained ResNet model
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # Remove the last layer (fully connected layer) of the ResNet
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.resnet.requires_grad_(False)
        # Define fully connected layers
        self.fc_layers = self._make_fc_layers()

    def forward(self, x):
        # Forward pass through ResNet
        x1 = x[:,:,:,:224]
        x2 = x[:,:,:,224:]

        x1 = self.resnet(x1)
        x2 = self.resnet(x2)

        x1 = x1.reshape(x1.size(0), -1)
        x2 = x2.reshape(x2.size(0), -1)
        
        # Concatenate x1 and x2
        x3 = torch.cat((x1, x2), dim=1)      
        # Forward pass through fully connected layers
        x3 = self.fc_layers(x3)
        
        return x3

    def _make_fc_layers(self):
        # Define sequential layers for fully connected layers
        return nn.Sequential(
            nn.Linear(4096, 2048),  # Adjust input dimension based on concatenated feature size
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

# Create an instance of the model
# net = Net()
#print(net)


def eval_model(model, data_loader, criterion, device,save=False):
    total_rmse = 0
    total_samples = 0
    val_loss = 0
    model.eval()
    outputsCon = list()

    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images = data['image']
            labels = data['label']
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if save:
                outputsCon.append(outputs)
            loss = criterion(outputs, labels.float())
            val_loss += loss.item()
            total_rmse += rmse(outputs, labels) * labels.size(0)
            total_samples += labels.size(0)

    average_rmse = total_rmse / total_samples
    if save:
        return val_loss / len(data_loader), average_rmse, outputsCon
    return val_loss / len(data_loader), average_rmse

def rmse(outputs, labels):
    return torch.sqrt(torch.mean((outputs - labels) ** 2))

def main():
    batch_size = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    
    transform_train = transforms.Compose([
        transforms.ToTensor()])

    transform_test = transforms.Compose([
        transforms.ToTensor()])

    dataset = VideoSoundDataset('./Dataset','../Audio/SampledData.csv',transform=None)
    train_dataset = Subset(dataset, range(0, 77069))
    testset = Subset(dataset, range(77069, len(dataset)))
    
    trainset, valset = torch.utils.data.random_split(train_dataset, [70000, 7069], generator=torch.Generator().manual_seed(0))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    learning_rate = 0.001
    l2_regularization = 0.0001
    epochs = 10

    net = Net().to(device)
    net = SiameseConvNet().to(device)
    # print(net) # print model architecture
    criterion = nn.MSELoss()  # Use Mean Squared Error loss for regression
    optimizer_adam = optim.Adam(net.parameters(), lr = learning_rate, weight_decay= l2_regularization)
    optimizer_sgd = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=l2_regularization)
    optimizer = optimizer_adam


    try:
        for epoch in range(epochs):
            running_loss = 0.0
            total_rmse = 0.0
            total_samples = 0
            net.train()
            for data in trainloader:
                inputs = data['image']
                labels = data['label']
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                outputs = net(inputs)
                outputs1 = torch.squeeze(outputs)
                loss = criterion(outputs1, labels.float())
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                total_rmse += rmse(outputs, labels) * labels.size(0)
                total_samples += labels.size(0)
                # _, predicted = torch.max(outputs.data, 1)
                # correct += (predicted == labels).sum().item()

            # if epoch % 10 == 0:
            #     val_loss, val_acc = eval_model(net, valloader, criterion, device)
            #     print('epoch - %d loss: %.3f accuracy: %.3f val_loss: %.3f val_acc: %.3f' % (epoch, running_loss / len(trainloader), 100 * correct / len(trainloader.dataset), val_loss, val_acc))
            # else:
            #     print('epoch - %d loss: %.3f accuracy: %.3f' % (epoch, running_loss / len(trainloader), 100 * correct / len(trainloader.dataset)))
            
            epoch_rmse = total_rmse / total_samples
            if epoch % 10 == 0:
                val_loss, val_rmse = eval_model(net, valloader, criterion, device)
                print(f'Epoch - {epoch} Loss: {running_loss / len(trainloader):.3f} RMSE: {epoch_rmse:.3f} Val Loss: {val_loss:.3f} Val RMSE: {val_rmse:.3f}')
            else:
                print(f'Epoch - {epoch} Loss: {running_loss / len(trainloader):.3f} RMSE: {epoch_rmse:.3f}')

        print('Finished training')
    except KeyboardInterrupt:
        pass

    try:
        net.eval()
        # Evaluate the model on the test set
        test_loss, test_acc, outputs = eval_model(net, testloader, criterion, device,save=True)
        concatenated_tensor = torch.cat(outputs, dim=0)
        concatenated_tensor_cpu = concatenated_tensor.cpu()
        torchaudio.save('output.wav', concatenated_tensor_cpu.unsqueeze(0), 2200)
        
    except:
        with open('tensor_data.txt', 'w') as f:
            concatenated_tensor_cpu = concatenated_tensor.cpu()
            f.write(str(concatenated_tensor_cpu.numpy()))

        # Alternatively torch.save for binary format which is more efficient
            torch.save(concatenated_tensor, 'tensor_data.pt')
    
    print('Test loss: %.3f accuracy: %.3f' % (test_loss, test_acc))



main()