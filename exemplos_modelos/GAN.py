import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

print(device)

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

class CatDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.cat_indices = [i for i, (img, label) in enumerate(dataset) if label == 3]

    def __getitem__(self, index):
        cat_index = self.cat_indices[index]
        return self.dataset[cat_index]

    def __len__(self):
        return len(self.cat_indices)

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
cat_dataset = CatDataset(train_dataset)
train_loader = DataLoader(cat_dataset, batch_size=128, shuffle=True)


def show_images(images):
    plt.figure(figsize=(8, 8))
    for i in range(images.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(np.transpose(images[i].detach().cpu().numpy(), (1, 2, 0)), interpolation='nearest')
        plt.axis('off')
    plt.show()
    
    
    
    
class Generator(nn.Module):
    """DCGAN Generator: maps latent noise to 3x32x32 images."""
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input: (100, 1, 1) -> (256, 4, 4)
            nn.ConvTranspose2d(100, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # (256, 4, 4) -> (128, 8, 8)
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # (128, 8, 8) -> (64, 16, 16)
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # (64, 16, 16) -> (3, 32, 32)
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )


    def forward(self, input):
        return self.main(input)



class Discriminator(nn.Module):
    """DCGAN Discriminator: binary classifier for real/fake images."""
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # Input: (3, 32, 32) -> (64, 16, 16)
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (64, 16, 16) -> (128, 8, 8)
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # (128, 8, 8) -> (256, 4, 4)
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # (256, 4, 4) -> (512, 2, 2)
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # Final projection to scalar score
            nn.Conv2d(512, 1, 2, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input).view(-1, 1).squeeze(1)
    
    
    
netG = Generator().to(device)
netD = Discriminator().to(device)

optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))




criterion = nn.BCELoss()

real_label = 1
fake_label = 0
num_epochs = 500

for epoch in range(num_epochs):
    for i, (data, _) in enumerate(train_loader):
        # Update D: maximize log(D(x)) + log(1 - D(G(z)))
        netD.zero_grad()
        real_cpu = data.to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)

        # Real batch pass
        output = netD(real_cpu)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # Fake batch pass
        noise = torch.randn(batch_size, 100, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        # Update G: maximize log(D(G(z)))
        netG.zero_grad()
        label.fill_(real_label)  # target real for generator objective
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        if i % 100 == 0:
            print(f'[{epoch}/{num_epochs}][{i}/{len(train_loader)}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}')

    # Save outputs
    if epoch % 2 == 0:
        f = (fake[:16].detach().cpu()+1)*0.5
        show_images(f)
        #save_image(fake.data, f'./data/fake_samples_epoch_{epoch}.png', normalize=True)

print("Training complete.")



num_epochs = 50

for epoch in range(num_epochs):
    for i, (data, _) in enumerate(train_loader):
        print(_)



netG = Generator().to(device)
test_noise = torch.randn(1, 100, 1, 1, device=device)
test_images = netG(test_noise)
print(test_images.size())  # Expected output: torch.Size([1, 3, 32, 32])




torch.save(netG.state_dict(), './netG.pth')
torch.save(netD.state_dict(), './netD.pth')



def generate_images(generator, num_images):
    with torch.no_grad():  # Temporarily set all the requires_grad flag to false
        noise = torch.randn(num_images, 100, 1, 1, device=device)  # 100 is the size of the noise vector
        generated_images = generator(noise)
        generated_images = (generated_images + 1) / 2  # Rescale images from [-1, 1] to [0, 1]
        return generated_images




images = generate_images(netG, 16)  # Generate 16 images
show_images(images)