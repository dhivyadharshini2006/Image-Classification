# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

Image classification is a fundamental problem in computer vision, where the goal is to assign an input image to one of the predefined categories. Traditional machine learning models rely heavily on handcrafted features, whereas Convolutional Neural Networks (CNNs) automatically learn spatial features directly from pixel data.

In this experiment, the task is to build a Convolutional Deep Neural Network (CNN) to classify images from the FashionMNIST dataset into their respective categories. The trained model will then be tested on new/unseen images to verify its effectiveness.

## Dataset
The FashionMNIST dataset consists of 70,000 grayscale images of size 28 × 28 pixels.

The dataset has 10 classes, representing different clothing categories such as T-shirt, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, and ankle boot.

Training set: 60,000 images.

Test set: 10,000 images.

Images are preprocessed (normalized and converted to tensors) before being passed into the CNN.
## Neural Network Model

<img width="962" height="468" alt="image" src="https://github.com/user-attachments/assets/9dd756d3-a1f3-48c3-b2bc-30ac4717949a" />

## DESIGN STEPS

### STEP 1:
Import the required libraries such as PyTorch, Torchvision, NumPy, and Matplotlib.

### STEP 2:
Load the FashionMNIST dataset and apply transformations (normalization, tensor conversion).

### STEP 3:
Split the dataset into training and testing sets.

### STEP 4:
Define the CNN architecture with convolutional, pooling, and fully connected layers.

### STEP 5:
Specify the loss function (CrossEntropyLoss) and optimizer (Adam).

### STEP 6:
Train the model using forward pass, loss computation, backpropagation, and parameter updates.

### STEP 7:
Evaluate the model on the test dataset and calculate accuracy.

### STEP 8:
Test the trained model on new/unseen FashionMNIST images.




## PROGRAM

### Name:Dhivya Dharshini B
### Register Number:212223240031
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
## Step 1: Load and Preprocess Data
# Define transformations for images
transform = transforms.Compose([
    transforms.ToTensor(),          # Convert images to tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize images
])


# Load Fashion-MNIST dataset
train_dataset = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=transform, download=True)
image, label = train_dataset[0]
print(image.shape)
print(len(train_dataset))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
class CNNClassifier (nn.Module):
  def __init__(self):
    super (CNNClassifier, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
    self.conv3 = nn. Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
    self.pool = nn. MaxPool2d(kernel_size=2, stride=2)
    self.fc1 = nn. Linear (128 * 3 * 3, 128)
    self.fc2 = nn. Linear (128, 64)
    self.fc3= nn. Linear (64, 10)
  def forward(self, x):
    x = self.pool (torch.relu(self.conv1(x)))
    x = self.pool (torch.relu(self.conv2(x)))
    x = self.pool(torch.relu(self.conv3(x)))
    x = x.view(x.size (0), -1) # Flatten the image
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = self.fc3(x)
    return x
image,label=train_dataset[0]
print(image.shape)
print(len(train_dataset))
class CNNClassifier (nn.Module):
  def __init__(self):
    super (CNNClassifier, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
    self.conv3 = nn. Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
    self.pool = nn. MaxPool2d(kernel_size=2, stride=2)
    self.fc1 = nn. Linear (128 * 3 * 3, 128)
    self.fc2 = nn. Linear (128, 64)
    self.fc3= nn. Linear (64, 10)
  def forward(self, x):
    x = self.pool (torch.relu(self.conv1(x)))
    x = self.pool (torch.relu(self.conv2(x)))
    x = self.pool(torch.relu(self.conv3(x)))
    x = x.view(x.size (0), -1) # Flatten the image
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = self.fc3(x)
    return x
from torchsummary import summary

# Initialize model
model = CNNClassifier()

# Move model to GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device)

# Print model summary
print("Name: Dhivya Dharshini B")
print("Register Number: 212223240031")
summary(model, input_size=(1, 28, 28))
model=CNNClassifier()
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)
def train_model(model,train_loader,criterion,optimizer,num_epochs=3):
  for epoch in range(num_epochs):
    running_loss=0.0
    for images,labels in train_loader:
      optimizer.zero_grad()
      outputs=model(images)
      loss=criterion(outputs,labels)
      loss.backward()
      optimizer.step()
      running_loss+=loss.item()
train_model(model,train_loader,criterion,optimizer)
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print('Name:        ')
    print('Register Number:       ')
    print(f'Test Accuracy: {accuracy:.4f}')

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    print('Name:        ')
    print('Register Number:       ')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Print classification report
    print('Name:Dhivya Dharshini B ')
    print('Register Number:212223240031      ')
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

     import matplotlib.pyplot as plt
def predict_image(model, image_index, dataset):
    model.eval()
    image, label = dataset[image_index]
    with torch.no_grad():
        output = model(image.unsqueeze(0))  # Add batch dimension
        _, predicted = torch.max(output, 1)
    class_names = dataset.classes

    # Display the image
    print('Name:Dhivya Dharshini B')
    print('Register Number:  212223240031')
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(f'Actual: {class_names[label]}\nPredicted: {class_names[predicted.item()]}')
    plt.axis("off")
    plt.show()
    print(f'Actual: {class_names[label]}, Predicted: {class_names[predicted.item()]}')
predict_image(model, image_index=80, dataset=test_dataset)
```

## OUTPUT
### Training Loss per Epoch
<img width="522" height="145" alt="image" src="https://github.com/user-attachments/assets/ff989f7c-e296-4992-92cc-d567dec1d3a5" />


### Confusion Matrix
<img width="709" height="608" alt="download" src="https://github.com/user-attachments/assets/95bb010e-4801-4475-8571-b8ebb22721dd" />


### Classification Report
<img width="626" height="431" alt="image" src="https://github.com/user-attachments/assets/65cea1dd-7514-47cd-8a26-fd2d28e5a964" />




### New Sample Data Prediction
<img width="551" height="619" alt="image" src="https://github.com/user-attachments/assets/563222df-6cbf-4558-96f4-40ddd0b77f55" />


## RESULT
Convolutional Deep Neural Network for Image Classification executed successfully.
