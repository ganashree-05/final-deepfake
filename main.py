import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
from sklearn.metrics import classification_report

# Define the model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 112 * 112, 2)  # 32 channels * 112 * 112 after pooling

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # Apply convolution and pool
        x = x.view(-1, 32 * 112 * 112)  # Flatten the output for the fully connected layer
        x = self.fc1(x)  # Fully connected layer
        return x

def preprocess_image(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file '{image_path}' not found!")
        return None

    image = Image.open(image_path).convert('RGB')  # Ensure 3 color channels (RGB)
    image = image.resize((224, 224))  # Resize to match model input

    # Convert image to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts shape to (C, H, W)
    ])
    image = transform(image)  # Shape will be (3, 224, 224)

    # Ensure it's float32 (Fix for type mismatch error)
    image = image.to(dtype=torch.float32)

    # Add batch dimension → final shape: (1, 3, 224, 224)
    image = image.unsqueeze(0)

    return image


# Dataset class to load images
class ImageDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        self.real_images = [os.path.join(real_dir, fname) for fname in os.listdir(real_dir)]
        self.fake_images = [os.path.join(fake_dir, fname) for fname in os.listdir(fake_dir)]
        self.images = self.real_images + self.fake_images
        self.labels = [0] * len(self.real_images) + [1] * len(self.fake_images)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

# Load dataset
def load_dataset(real_dir, fake_dir):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = ImageDataset(real_dir, fake_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader

# Save the trained model
def save_model(model, save_path='saved_models/trained_model.pth'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure the directory exists
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

from torchvision.models import resnet18, ResNet18_Weights

def load_model(model_path='saved_models/trained_model.pth'):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)  # Use new weights method
    model.fc = nn.Linear(512, 2)

    if not os.path.exists(model_path):
        print(f"❌ Error: Model file '{model_path}' not found! Train the model first.")
        return None

    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"✅ Model loaded from {model_path}")
    return model



def predict_image(image_path):
    model = load_model()  # Load trained model

    if model is None:
        return "Error: Model not found!"  # Return error message

    # Preprocess the image
    image = preprocess_image(image_path)
    if image is None:
        return "Error: Image not found!"  # Return error message

    # Get the model's prediction
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)

    # Return prediction result
    return "Real" if preds.item() == 0 else "Fake"

def train_model():
        real_dir = "data/training_real"
        fake_dir = "data/training_fake"
        train_loader, test_loader = load_dataset(real_dir, fake_dir)

        model = resnet18(weights=ResNet18_Weights.DEFAULT)  # Update weights method
        model.fc = nn.Linear(512, 2)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        num_epochs = 30  # Increased epochs
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

        os.makedirs('saved_models', exist_ok=True)
        torch.save(model.state_dict(), "saved_models/trained_model.pth")
        print("✅ Model trained and saved.")


# Main function
if __name__ == "__main__":
    print("Starting training...")
    train_model()  # Train and save the model

    # Predict a new image
    image_path = "predictions/img_1.png"  # <-- Change this to the path of the image you want to predict
    predict_image(image_path)  # Call the prediction function
