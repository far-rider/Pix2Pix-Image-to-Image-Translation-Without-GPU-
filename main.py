import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image

# ✅ Fix 1: Correct Dataset Class for Paired Images
class FacadesDataset(Dataset):
    def __init__(self, root, mode="train", transform=None):
        self.transform = transform
        self.image_dir = os.path.join(root, mode)
        self.image_filenames = sorted(os.listdir(self.image_dir))  # List images
        
    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        
        image = Image.open(image_path).convert("RGB")  # Load image
        width, height = image.size

        # ✅ Fix 2: Split image into input and target
        input_image = image.crop((0, 0, width // 2, height))  # Left half
        target_image = image.crop((width // 2, 0, width, height))  # Right half

        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return input_image, target_image

# ✅ Fix 3: Correct Normalization for RGB Images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ✅ Load dataset
dataset = FacadesDataset(root="data/facades", mode="train", transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# ✅ Debugging: Print shape of one batch
for i, (input_image, target_image) in enumerate(dataloader):
    print(f"Batch {i+1} - Input Shape: {input_image.shape}, Target Shape: {target_image.shape}")
    break  # Stop after printing one batch

# ✅ Define Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# ✅ Define Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        return self.model(torch.cat((x, y), dim=1))

# ✅ Initialize models
device = torch.device("cpu")
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# ✅ Loss and Optimizers
adversarial_loss = nn.BCELoss()
l1_loss = nn.L1Loss()
gen_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
disc_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)

# ✅ Training Loop
for epoch in range(5):
    for i, (input_image, target_image) in enumerate(dataloader):
        input_image, target_image = input_image.to(device), target_image.to(device)

        # Train Generator
        gen_optimizer.zero_grad()
        fake_image = generator(input_image)

        fake_pred = discriminator(input_image, fake_image)
        gen_loss = adversarial_loss(fake_pred, torch.ones_like(fake_pred)) + l1_loss(fake_image, target_image)
        gen_loss.backward()
        gen_optimizer.step()

        # Train Discriminator
        disc_optimizer.zero_grad()
        real_pred = discriminator(input_image, target_image)
        fake_pred = discriminator(input_image, fake_image.detach())
        disc_loss = adversarial_loss(real_pred, torch.ones_like(real_pred)) + adversarial_loss(fake_pred, torch.zeros_like(fake_pred))
        disc_loss.backward()
        disc_optimizer.step()

        print(f"Epoch {epoch+1}, Step {i+1}, Gen Loss: {gen_loss.item():.4f}, Disc Loss: {disc_loss.item():.4f}")

import matplotlib.pyplot as plt

# Function to generate and display an image
def generate_image(input_image):
    generator.eval()  # Set generator to evaluation mode
    with torch.no_grad():
        fake_image = generator(input_image)
    return fake_image.squeeze().permute(1, 2, 0)  # Convert tensor to image format

# Select a sample input image
sample_image, _ = dataset[0]
generated_image = generate_image(sample_image.unsqueeze(0))

# Display original and generated images
plt.subplot(1, 2, 1)
plt.imshow(sample_image.permute(1, 2, 0))
plt.title("Input Image")

plt.subplot(1, 2, 2)
plt.imshow(generated_image)
plt.title("Generated Image")

plt.show() 