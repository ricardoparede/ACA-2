import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

print(device)


transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
mnist_data_flattened = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

data_loader_flattened = torch.utils.data.DataLoader(dataset=mnist_data_flattened,
                                          batch_size=64,
                                          shuffle=True)


dataiter_flattened = iter(data_loader_flattened)
images, labels = next(dataiter_flattened)
print(torch.min(images), torch.max(images))



class Autoencoder_Linear(nn.Module):
    def __init__(self, color_channels=1, width=28, height=28):
        super().__init__()        
        self.target_width = width
        self.target_height = height
        self.target_color_channels = color_channels
        self.encoder = nn.Sequential(
            nn.Linear(self.target_color_channels * self.target_height * self.target_width, 128), # (BatchSize, Width*Height) -> (BatchSize, 128)
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3)
        )
        

        self.decoder = nn.Sequential(
            #TODO: Implement the decoder part of the autoencoder
            #Remember that the activation function of the last layer 
            #depends on the range of the input values
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    




def train(model, data_loader, num_epochs=50, device='cpu'):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=1e-3, 
                                weight_decay=1e-5)


    model = model.to(device)
    
    outputs = []
    for epoch in range(num_epochs):
        for (img, _) in data_loader:
            #TODO: Implement the training loop
            #Remember to move the data to the device and that
            #the training loop should be similar to the one of
            #a regular neural network
            pass
            
            
        print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')
        outputs.append((epoch, img.to('cpu'), recon.to('cpu')))
        
    
    return model.to('cpu'), outputs




num_epochs = 5
linear_model = Autoencoder_Linear()
linear_model, outputs = train(linear_model, data_loader_flattened, num_epochs=num_epochs, device=device)




def plot_results(outputs, num_epochs, color_channels=1, width=28, height=28):
    for k in range(0, num_epochs, 1):
        plt.figure(figsize=(9, 2))
        plt.gray()
        imgs = outputs[k][1].detach().numpy()
        recon = outputs[k][2].detach().numpy()

        
        for i, item in enumerate(imgs):
            if i >= 9: break
            plt.subplot(2, 9, i+1)
            plt.axis('off')

            if color_channels == 1:
                plt.imshow(item.reshape(height, width))
            else:
                item = item.reshape(color_channels,height, width)
                plt.imshow(np.transpose(item, (1,2,0)))
    
        for i, item in enumerate(recon):
            if i >= 9: break
            plt.subplot(2, 9, 9+i+1)
            
            plt.axis('off')
            if color_channels == 1:
                plt.imshow(item.reshape(height, width))
            else:
                item = item.reshape(color_channels,height, width)
                plt.imshow(np.transpose(item, (1,2,0)))




print(len(outputs[0][1]))
plot_results(outputs, num_epochs, color_channels=1, width=28, height=28)




transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view( -1))])
data_flattened = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

data_loader_flattened = torch.utils.data.DataLoader(dataset=data_flattened,
                                          batch_size=64,
                                          shuffle=True)

dataiter_flattened = iter(data_loader_flattened)
images, labels = next(dataiter_flattened)
print(torch.min(images), torch.max(images))
print(images.shape)



color_channels = 3
width = 32
height = 32
linear_model = Autoencoder_Linear( color_channels=3, width=32, height=32)
linear_model, outputs = train(linear_model, data_loader_flattened, num_epochs=num_epochs, device=device)

plot_results(outputs, num_epochs, color_channels, width, height)




class Autoencoder(nn.Module):
    def __init__(self, color_channels=1, width=32, height=32):
        super().__init__()        
        self.encoder = nn.Sequential(
            nn.Conv2d(color_channels, 16, 3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 8)
        )
        
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 8),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, color_channels, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
 
 
 
 
transform = transforms.ToTensor()
data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

data_loader = torch.utils.data.DataLoader(dataset=data,
                                          batch_size=64,
                                          shuffle=True)

dataiter = iter(data_loader)
images, labels = next(dataiter)
print(torch.min(images), torch.max(images))
print(images.shape)




color_channels = 3
width = 32
height = 32
model = Autoencoder(3, 32, 32)
model, outputs = train(model, data_loader, num_epochs=num_epochs, device=device)



plot_results(outputs, num_epochs, color_channels, width, height)




class AutoencoderTweaked(nn.Module):
    pass
