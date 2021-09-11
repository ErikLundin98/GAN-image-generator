import torch
import numpy as np

from torchvision import transforms, datasets
from torch.utils.data import ConcatDataset

import os

import matplotlib.pyplot as plt
# File to run when the dataset is processed
# hyperparams
SEED_SIZE = 16 # latent variable
DEVICE = 'cpu'
BATCH_SIZE = 32
IMAGE_SIZE = 64

class PadToSize:
	def __init__(self, width, height):
		self.width = width
		self.height = height
	def __call__(self, image):
		h = image.shape[1]
		w = image.shape[2]
		diff_h = self.height-h
		diff_w = self.width-w
		ph1 = diff_h//2
		ph2 = diff_h - ph1
		pw1 = diff_w//2
		pw2 = diff_w - pw1
		#image = np.where(image==0, 255, image)
		return torch.from_numpy(np.pad(image, pad_width=((0, 0),(ph1, ph2),(pw1, pw2)), mode='constant', constant_values=0)) # whiten

data_folder = os.path.join(os.getcwd(),'raw_dataset')
dataset = datasets.ImageFolder(data_folder, transform=transforms.Compose([
    transforms.ToTensor(),
    PadToSize(IMAGE_SIZE, IMAGE_SIZE),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    
]))
mirrored_dataset = datasets.ImageFolder(data_folder, transform=transforms.Compose([
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.ToTensor(),
    PadToSize(IMAGE_SIZE, IMAGE_SIZE),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    
]))
color_jittered_dataset = datasets.ImageFolder(data_folder, transform=transforms.Compose([
    transforms.ToTensor(),
    PadToSize(IMAGE_SIZE, IMAGE_SIZE),
    transforms.ColorJitter(0.5, 0.5, 0.5),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    
]))
color_jitter_flipped_dataset = datasets.ImageFolder(data_folder, transform=transforms.Compose([
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.ToTensor(),
    PadToSize(IMAGE_SIZE, IMAGE_SIZE),
    transforms.ColorJitter(0.5, 0.5, 0.5),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    
]))

combined = ConcatDataset([dataset, mirrored_dataset, color_jittered_dataset, color_jitter_flipped_dataset])

dataloader = torch.utils.data.DataLoader(combined, BATCH_SIZE, shuffle=True)

def normalize(data):
    return (data - np.min(data)) / np.ptp(data)

def show(img, save=False, path=None):
    if type(img) is not np.ndarray:
        img = img.numpy()
    img = normalize(img)
    plt.imshow(np.transpose(img, (1,2,0)), vmin=-1, vmax=1)
    if save:
        plt.savefig(path)

class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.input_stack = torch.nn.Sequential(
            torch.nn.Conv2d(3, IMAGE_SIZE, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(0.2, inplace=True),
        ) # 64x32x32
        self.stack0 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.2, inplace=True),
        ) # 128x16x16
        self.stack1 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.2, inplace=True),
        ) # 128x8x8
        self.stack2 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.2, inplace=True),
        ) # 128x4x4
        self.output = torch.nn.Sequential(
            torch.nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=0, bias=False),
            torch.nn.Flatten(),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.input_stack(x)
        x = self.stack0(x)
        x = self.stack1(x)
        x = self.stack2(x)
        x = self.output(x)
        return x


class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.input_stack = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(SEED_SIZE, 128, kernel_size=4, padding=0, stride=1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(True),
        ) # 64x32x32
        self.stack0 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 128, kernel_size=4, padding=1, stride=2, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(True),
        ) # 128x16x16
        self.stack1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 128, kernel_size=4, padding=1, stride=2, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(True),
        ) # 128x8x8
        self.stack2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 64, kernel_size=4, padding=1, stride=2, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(True),
        ) # 128x4x4
        self.output = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 3, kernel_size=4, padding=1, stride=2, bias=False),
            torch.nn.Tanh()
        )

    def forward(self, x):
        x = self.input_stack(x)
        x = self.stack0(x)
        x = self.stack1(x)
        x = self.stack2(x)
        x = self.output(x)
        return x


def train_discriminator(discriminator, generator, real_pokemon, optimizer):
    optimizer.zero_grad()
    real_predictions = discriminator(real_pokemon)
    real_labels = torch.rand(real_pokemon.size(0), 1, device=DEVICE)*0.1
    real_loss = torch.nn.functional.binary_cross_entropy(real_predictions, real_labels)
    real_score = torch.mean(real_predictions).item()

    latent_seeds = torch.randn(BATCH_SIZE, SEED_SIZE, 1, 1, device=DEVICE)
    fake_pokemon = generator(latent_seeds)
    fake_predictions = discriminator(fake_pokemon)
    fake_labels = torch.rand(fake_pokemon.size(0), 1, device=DEVICE)*0.1 + 0.9
    fake_loss = torch.nn.functional.binary_cross_entropy(fake_predictions, fake_labels)
    fake_score = torch.mean(fake_predictions).item()

    total_loss = real_loss + fake_loss
    total_loss.backward()
    optimizer.step()

    return total_loss.item(), real_score, fake_score

def train_generator(generator, discriminator, optimizer):
   
    optimizer.zero_grad()
    
    latent_seeds = torch.randn(BATCH_SIZE, SEED_SIZE, 1, 1, device=DEVICE)
    fake_pokemon = generator(latent_seeds)

    discriminator_predictions = discriminator(fake_pokemon)
    labels = torch.zeros(fake_pokemon.size(0), 1, device=DEVICE) 
    loss = torch.nn.functional.binary_cross_entropy(discriminator_predictions, labels)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()


def save_results(epoch, seed, generator, display=False):
    generated = generator(seed)
    images = [generated[i].detach() for i in range(len(generated))]
    concatenated_images = np.concatenate(images, axis=2)
    show(concatenated_images, save=True, path='progression/{}.png'.format(epoch))


def train(epochs, learning_rate, discriminator, generator, reference_seed, start_epoch=0):
    # Track losses and scores
    disc_losses = []
    disc_scores = []
    gen_losses = []
    gen_scores = []
    
    # Create the optimizers
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.9))
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.9))
    
    # Run the loop
    for epoch in range(start_epoch, epochs):
        # Go through each image
        for batch, (real_images, _) in enumerate(dataloader):
            # Train the discriminator
            disc_loss, real_score, gen_score = train_discriminator(discriminator, generator, real_images, disc_optimizer)

            # Train the generator
            gen_loss = train_generator(generator, discriminator, gen_optimizer)
        
        # Collect results
        disc_losses.append(disc_loss)
        disc_scores.append(real_score)
        gen_losses.append(gen_loss)
        gen_scores.append(gen_score)
        
        # Print the losses and scores
        print("Epoch [{}/{}], gen_loss: {:.4f}, disc_loss: {:.4f}, real_score: {:.4f}, gen_score: {:.4f}".format(
            epoch, epochs, gen_loss, disc_loss, real_score, gen_score))
        
        # Save model
        if epoch % 10 == 0:
            torch.save(discriminator.state_dict(), 'models/discriminator.pth')
            torch.save(generator.state_dict(), 'models/generator.pth')
        # Save the images and show the progress
        save_results(epoch=epoch, seed=reference_seed, generator=generator, display=True)
    
    # Return stats
    return disc_losses, disc_scores, gen_losses, gen_scores


# Use this for new model training with intel gpu

discriminator = Discriminator().to(DEVICE)
generator = Generator().to(DEVICE)
train(epochs=100, 
learning_rate=0.01, 
discriminator=discriminator, 
generator=generator, 
reference_seed=torch.randn(8, SEED_SIZE, 1, 1, device=DEVICE),
start_epoch=0)