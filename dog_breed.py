import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import os
from PIL import Image

import torch
from torchsummary import summary
from torch import nn, optim
from torch.functional import F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.autograd import Variable

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor']= 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['font.size']=12

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_dir = './input/dog-breed-identification/'
labels = pd.read_csv(os.path.join(data_dir, 'labels.csv'))
assert(len(os.listdir(os.path.join(data_dir, 'train'))) == len(labels))

print(labels.head())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
labels.breed = le.fit_transform(labels.breed)
labels.head()

print(labels.head())

X = labels.id
y = labels.breed

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y,test_size=0.4, random_state=SEED, stratify=y)
X_valid, X_test, y_valid, y_test = train_test_split(X_valid, y_valid, test_size=0.5, random_state=SEED, stratify=y_valid)

class Dataset_Interpreter(Dataset):
    def __init__(self, data_path, file_names, labels=None, transforms=None):
        self.data_path = data_path
        self.file_names = file_names
        self.labels = labels
        self.transforms = transforms
        
    def __len__(self):
        return (len(self.file_names))
    
    def __getitem__(self, idx):
        img_name = f'{self.file_names.iloc[idx]}.jpg'
        full_address = os.path.join(self.data_path, img_name)
        image = Image.open(full_address)
        label = self.labels.iloc[idx]
        
        if self.transforms is not None:
            image = self.transforms(image)
            
        return np.array(image), label
    
def plot_images(images):

    n_images = len(images)

    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    fig = plt.figure(figsize=(20,10))
    for i in range(rows*cols):
        ax = fig.add_subplot(rows, cols, i+1)
        ax.set_title(f'{le.inverse_transform([images[i][1]])}')
        ax.imshow(np.array(images[i][0]))
        ax.axis('off')
        
N_IMAGES = 9

train_data = Dataset_Interpreter(data_path=data_dir+'train/', file_names=X_train, labels=y_train)
images = [(image, label) for image, label in [train_data[i] for i in range(N_IMAGES)]] 
plot_images(images)

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
train_transforms = transforms.Compose([transforms.Resize(32),
                               transforms.CenterCrop(32),
                               transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
                               transforms.RandomHorizontalFlip(p=0.5),
                               transforms.RandomVerticalFlip(p=0.5),
                               transforms.RandomGrayscale(p=0.1), 
                               transforms.ToTensor(),
                               normalize])
test_transforms = transforms.Compose([transforms.Resize(32),
                               transforms.CenterCrop(32),
                               transforms.ToTensor(),
                               normalize])

train_data = Dataset_Interpreter(data_path=data_dir+'train/', file_names=X_train, labels=y_train, transforms=train_transforms)
valid_data = Dataset_Interpreter(data_path=data_dir+'train/', file_names=X_valid, labels=y_valid, transforms=test_transforms)
test_data = Dataset_Interpreter(data_path=data_dir+'train/', file_names=X_test, labels=y_test, transforms=test_transforms)

print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
print(f'Number of testing examples: {len(test_data)}')

BATCH_SIZE = 64

train_iterator = DataLoader(train_data, shuffle=True, batch_size= BATCH_SIZE)
valid_iterator = DataLoader(valid_data, batch_size=BATCH_SIZE)
test_iterator = DataLoader(test_data, batch_size = BATCH_SIZE)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5), #stride=1, padding=0 is a default
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 120)   #num_classes = 120
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.features(x)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        
        return x
model = LeNet().to(device)
summary(model, (3, 32, 32))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model= LeNet().to(device)
loss_criterion = nn.CrossEntropyLoss().to(device)
optimizer=optim.Adam(model.parameters())

print(f'The model has {count_parameters(model):,} trainable parameters')

def train(model, iterator, optimizer, criterion, device):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for (x, y) in iterator:
        
        x = Variable(torch.FloatTensor(np.array(x))).to(device)
        y = Variable(torch.LongTensor(y)).to(device)
        
        optimizer.zero_grad()
                
        y_pred = model(x)
        
        loss = criterion(y_pred, y)
        
        acc = calculate_accuracy(y_pred, y)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, device):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
        
        for (x, y) in iterator:

            x = Variable(torch.FloatTensor(np.array(x))).to(device)
            y = Variable(torch.LongTensor(y)).to(device)
        
            y_pred = model(x)

            loss = criterion(y_pred, y)

            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def fit_model(model, model_name, train_iterator, valid_iterator, optimizer, loss_criterion, device, epochs):
    """ Fits a dataset to model"""
    best_valid_loss = float('inf')
    
    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []
    
    for epoch in range(epochs):
    
        start_time = time.time()
    
        train_loss, train_acc = train(model, train_iterator, optimizer, loss_criterion, device)
        valid_loss, valid_acc = evaluate(model, valid_iterator, loss_criterion, device)
        
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_accs.append(train_acc*100)
        valid_accs.append(valid_acc*100)
    
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f'{model_name}.pt')
    
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
        
    return pd.DataFrame({f'{model_name}_Training_Loss':train_losses, 
                        f'{model_name}_Training_Acc':train_accs, 
                        f'{model_name}_Validation_Loss':valid_losses, 
                        f'{model_name}_Validation_Acc':valid_accs})

train_stats_LeNet = fit_model(model, 'LeNet', train_iterator, valid_iterator, optimizer, loss_criterion, device, epochs=20)

def plot_training_statistics(train_stats, model_name):
    
    fig, axes = plt.subplots(2, figsize=(15,15))
    axes[0].plot(train_stats[f'{model_name}_Training_Loss'], label=f'{model_name}_Training_Loss')
    axes[0].plot(train_stats[f'{model_name}_Validation_Loss'], label=f'{model_name}_Validation_Loss')
    axes[1].plot(train_stats[f'{model_name}_Training_Acc'], label=f'{model_name}_Training_Acc')
    axes[1].plot(train_stats[f'{model_name}_Validation_Acc'], label=f'{model_name}_Validation_Acc')
    
    axes[0].set_xlabel("Number of Epochs"), axes[0].set_ylabel("Loss")
    axes[1].set_xlabel("Number of Epochs"), axes[1].set_ylabel("Accuracy in %")
    
    axes[0].legend(), axes[1].legend()

plot_training_statistics(train_stats_LeNet, 'LeNet')


###### TRAINING ON RESNET-18 WITH TRANSFER LEARNING ######
from torchvision import models
model = models.resnet18(pretrained=True).to(device)
print(model)

for name, param in model.named_parameters():
    if("bn" not in name):
        param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features,120).to(device)
optimizer = optim.Adam(model.parameters(), lr = 1e-2)

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
train_transforms = transforms.Compose([transforms.Resize(224),
                               transforms.CenterCrop(224),
                               transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
                               transforms.RandomHorizontalFlip(p=0.5),
                               transforms.RandomVerticalFlip(p=0.5),
                               transforms.RandomGrayscale(p=0.1), 
                               transforms.ToTensor(),
                               normalize])
test_transforms = transforms.Compose([transforms.Resize(224),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               normalize])

train_data = Dataset_Interpreter(data_path=data_dir+'train/', file_names=X_train, labels=y_train, transforms=train_transforms)
valid_data = Dataset_Interpreter(data_path=data_dir+'train/', file_names=X_valid, labels=y_valid, transforms=test_transforms)
test_data = Dataset_Interpreter(data_path=data_dir+'train/', file_names=X_test, labels=y_test, transforms=test_transforms)

BATCH_SIZE = 64

train_iterator = DataLoader(train_data, shuffle=True, batch_size= BATCH_SIZE)
valid_iterator = DataLoader(valid_data, batch_size=BATCH_SIZE)
test_iterator = DataLoader(test_data, batch_size = BATCH_SIZE)

train_stats_ResNet18 = fit_model(model, 'ResNet18', train_iterator, valid_iterator, optimizer, loss_criterion, device, epochs=20)

plot_training_statistics(train_stats_ResNet18, 'ResNet18')


#### TRAINING ON RESNET-34 WITH TRANSFER LEARNING ####
from torchvision import models
model = models.resnet34(pretrained=True).to(device)
print(model)

for name, param in model.named_parameters():
    if("bn" not in name):
        param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features,120).to(device)
optimizer = optim.Adam(model.parameters(), lr = 1e-2)

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
train_transforms = transforms.Compose([transforms.Resize(224),
                               transforms.CenterCrop(224),
                               transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
                               transforms.RandomHorizontalFlip(p=0.5),
                               transforms.RandomVerticalFlip(p=0.5),
                               transforms.RandomGrayscale(p=0.1), 
                               transforms.ToTensor(),
                               normalize])
test_transforms = transforms.Compose([transforms.Resize(224),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               normalize])

train_data = Dataset_Interpreter(data_path=data_dir+'train/', file_names=X_train, labels=y_train, transforms=train_transforms)
valid_data = Dataset_Interpreter(data_path=data_dir+'train/', file_names=X_valid, labels=y_valid, transforms=test_transforms)
test_data = Dataset_Interpreter(data_path=data_dir+'train/', file_names=X_test, labels=y_test, transforms=test_transforms)

BATCH_SIZE = 64

train_iterator = DataLoader(train_data, shuffle=True, batch_size= BATCH_SIZE)
valid_iterator = DataLoader(valid_data, batch_size=BATCH_SIZE)
test_iterator = DataLoader(test_data, batch_size = BATCH_SIZE)

train_stats_ResNet34 = fit_model(model, 'ResNet34', train_iterator, valid_iterator, optimizer, loss_criterion, device, epochs=20)