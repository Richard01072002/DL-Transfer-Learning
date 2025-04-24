# DL- Developing a Neural Network Classification Model using Transfer Learning

## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## THEORY
A Convolutional Neural Network (CNN) is a type of deep learning model designed to process and classify visual data. It uses convolutional layers to automatically learn spatial features from images, followed by pooling layers to reduce dimensionality and fully connected layers for classification. CNNs are especially effective for image recognition tasks due to their ability to capture patterns like edges, textures, and shapes. In this project, a CNN was used to classify images of cats and dogs by learning features directly from the input images without manual feature extraction.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 
1. Data Preparation
-Apply data augmentation and normalization using transforms.Compose.
-Load training and testing datasets with ImageFolder from a specified directory (CATS_DOGS/train and CATS_DOGS/test).
-Use DataLoader to create iterable batches with shuffling for training.

### STEP 2: 
2 convolutional layers
-Max pooling
-Fully connected layers
-Log softmax for the final output
-Print the number of parameters to verify model complexity.


### STEP 3: 
3. Training Setup
-Define loss function: CrossEntropyLoss
-Optimizer: Adam
-Set seed for reproducibility and define training hyperparameters (epochs, batch limits).


### STEP 4: 
4. Training & Evaluation Loop
-For each epoch:
-Train: Forward pass → compute loss → backpropagate → optimize
-Evaluate: Disable gradient calculation, compute test accuracy/loss
-Log results periodically.


### STEP 5: 
5. Results & Saving
-Save the trained model using torch.save().
-Plot:
-Training & validation loss
-Training & validation accuracy
-Display final test accuracy as a percentage.


## PROGRAM

### Name:RICHARDSON A

### Register Number:212222233005

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models # add models to the list
from torchvision.utils import make_grid
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

# ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")


train_transform = transforms.Compose([
        transforms.RandomRotation(10),      # rotate +/- 10 degrees
        transforms.RandomHorizontalFlip(),  # reverse 50% of images
        transforms.Resize(224),             # resize shortest side to 224 pixels
        transforms.CenterCrop(224),         # crop longest side to 224 pixels at center
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

import os
print(os.getcwd())


#../Data/CATS_DOGS
root = 'CATS_DOGS'

train_data = datasets.ImageFolder(os.path.join(root, 'train'), transform=train_transform)

test_data = datasets.ImageFolder(os.path.join(root, 'test'), transform=test_transform)

torch.manual_seed(42)
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10)

class_names = train_data.classes

class_names
len(train_data)
len(test_data)

# Grab the first batch of 10 images
for images,labels in train_loader: 
    break

images.shape   # 10 images, 3 colour channel , each channel is 224 x 224

# to display 
im = make_grid(images, nrow=5)  # the default nrow is 8

# Inverse normalize the images
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)
im_inv = inv_normalize(im)

plt.figure(figsize=(12,4))
plt.imshow(np.transpose(im_inv.numpy(), (1, 2, 0)));
plt.show()

class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(54*54*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 54*54*16)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)

(((224-2)/2)-2)/2

torch.manual_seed(101)
CNNmodel = ConvolutionalNetwork()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(CNNmodel.parameters(), lr=0.0001)

CNNmodel

for p in CNNmodel.parameters():
    print(p.numel())

print("RICHARDSON A")

# Training the mmodel

import time
start_time = time.time()



epochs = 3

# optional limits on number of batches
max_trn_batch = 800    # batch 10 images ---> 8000 images
max_tst_batch = 300    # 300 max images 

train_losses = []
test_losses = []
train_correct = []
test_correct = []

for i in range(epochs):
    trn_corr = 0
    tst_corr = 0
    
    # Run the training batches
    for b, (X_train, y_train) in enumerate(train_loader):
        
        # optional Limit the number of batches
        if b == max_trn_batch:
            break
        b+=1
        
        # Apply the model
        y_pred = CNNmodel(X_train)
        loss = criterion(y_pred, y_train)
 
        # Tally the number of correct predictions
        predicted = torch.max(y_pred.data, 1)[1]
        batch_corr = (predicted == y_train).sum()
        trn_corr += batch_corr
        
        # Update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print interim results
        if b%200 == 0:
            print(f'Epoch: {i}  loss: {loss.item()}')

    train_losses.append(loss)
    train_correct.append(trn_corr)

    # Run the testing batches
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):
            # optional Limit the number of batches
            if b == max_tst_batch:
                break

            # Apply the model
            y_val = CNNmodel(X_test)

            # Tally the number of correct predictions
            predicted = torch.max(y_val.data, 1)[1] 
            batch_corr = (predicted == y_test).sum()
            tst_corr += tst_corr + batch_corr

    loss = criterion(y_val, y_test)
    test_losses.append(loss)
    test_correct.append(tst_corr)

    
total_time = time.time() - start_time
print("RICHARDSON A")
print(f'Total Time: {total_time/60} minutes') # print the time elapsed

torch.save(CNNmodel.state_dict(), 'myImageCNNModel.pt')

train_losses = [float(loss.detach().numpy()) if torch.is_tensor(loss) else float(loss) for loss in train_losses]
test_losses = [float(loss.detach().numpy()) if torch.is_tensor(loss) else float(loss) for loss in test_losses]

plt.plot(train_losses, label='training loss')
plt.plot(test_losses, label='validation loss')
plt.title('Loss at the end of each epoch\n BY RICHARDSON A')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

plt.plot([t / 80 for t in train_correct], label='Training Accuracy')
plt.plot([t / 30 for t in test_correct], label='Validation Accuracy')
plt.title('Accuracy at the End of Each Epoch\n BY RICHARDSON A')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

test_correct[-1].item()/3000
100*test_correct[-1].item()/3000

```

### OUTPUT

## Training Loss, Validation Loss Vs Iteration Plot

<img width="582" alt="Screenshot 2025-04-24 at 2 22 44 PM" src="https://github.com/user-attachments/assets/681ddd5f-2ee2-4770-807b-b8d4d763234e" />

## to display Inverse normalize the images
<img width="838" alt="Screenshot 2025-04-24 at 2 26 36 PM" src="https://github.com/user-attachments/assets/fba85c10-592c-4dcf-8853-e11b71d3d5c7" />


## CNNmodel
<img width="353" alt="Screenshot 2025-04-24 at 2 27 28 PM" src="https://github.com/user-attachments/assets/ccb00e9f-901e-4139-8ef9-6318b4819f5a" />

### Training the mmodel
<img width="537" alt="image" src="https://github.com/user-attachments/assets/be7cea8e-11ad-4875-915d-be3b29fa0da8" />

## test correct 
<img width="202" alt="Screenshot 2025-04-24 at 2 29 00 PM" src="https://github.com/user-attachments/assets/afc78425-87ac-4fe7-8fbb-6b1d00b94c13" />
<img width="241" alt="Screenshot 2025-04-24 at 2 29 12 PM" src="https://github.com/user-attachments/assets/2ac59a09-80d5-484f-ab40-23bf92ffeeb2" />

## RESULT
After training the custom CNN model for 3 epochs on the Cats vs Dogs dataset, we achieved a final test accuracy of approximately 87.33%, with a training loss of around 0.4268 and a validation loss of 0.5372. The model showed good learning progress over epochs, and the results indicate that it can effectively distinguish between cat and dog images. The trained model was saved for future use.
