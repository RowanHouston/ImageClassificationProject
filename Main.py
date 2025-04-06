import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import git

import shutil
import random

classes = ["lymphocyte", "eosinophil", "erythroblast", "ig", "monocyte", "neutrophil", "platelet" , "basophil"]

#device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def class_to_idx(label):
    # classes = ("lymphocyte", "basophil", "eosinophil", "erythroblast", "ig", "monocyte", "neutrophil", "platelet")
    label_list = torch.Tensor([[0, 0, 0, 0, 0, 0, 0, 0]])
    label_list[0][classes.index(label)] = 1
    return label_list
    # label_list = torch.Tensor([[0, 0, 0, 0, 0, 0, 0, 0]])
    # for i in range(len(classes)):
    #     if classes[0][i] == label:
    #         label_list[0][i] = 1
    # return label_list


def idx_to_class(idx):
    for i in range(len(idx[0])):
        if idx[0][i] == 1:
            return torch.tensor([i])
    



#Dataset Class 

class bloodcells_dataset:
    def __init__(self, image_paths, transform = None):
        self.image_paths = image_paths
        self.transform = transform # transform the dataset given a transformation
        
        
    def __len__(self): # gets the size of the dataset
        return len(self.image_paths)
    def __getitem__(self, index): #returns one datapoint at a time
        image_filepath = self.image_paths[index]
        image = Image.open(image_filepath).convert('RGB')
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((359, 359))
        ])
        image = transform(image)

        # image = cv2.imread(image_filepath)
        # image = image[0:359, 0:359]
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = transforms.ToTensor(np.array(image))
        
        label = (image_filepath.split("\\"))[-2]
        label = class_to_idx(label)
        if self.transform != None:
            image = self.transform(image = image)["image"] # our transforms when called must return the transformed image 
            
        return image, label
    
    

# classes = ("lymphocyte", "eosinophil", "erythroblast", "ig", "monocyte", "neutrophil", "platelet" , "basophil", )

dataset = "c:/Users/tilli/OneDrive/Desktop/ImageClassificationProject/bloodcells_dataset"
#print(os.listdir("c:/Users/tilli/OneDrive/Desktop/ImageClassificationProject/bloodcells_dataset/basophil"))

# make our train and test datasets: 
# These datasets should be random but different from eachother?


def make_sets(dataset, classes): # copy random files into the training dataset
    for i in range (len(classes)):
        os.makedirs("c:/Users/tilli/OneDrive/Desktop/ImageClassificationProject/Training/" + str(classes[i]))  # list of files function works
        os.makedirs("c:/Users/tilli/OneDrive/Desktop/ImageClassificationProject/Testing/" + str(classes[i]))
        # make the file paths
        # add random files to new training file path
        class_files = os.listdir(dataset + "\\" + str(classes[i]))
        c_files_temp = class_files.copy()
        #print(class_files)
        for j in range (round(len(class_files) * 0.8)):
            
            rand_file_index = random.randint(0, len(c_files_temp) - 1)
            
            shutil.copy(dataset + "\\" + str(classes[i]) + "\\"+ str(c_files_temp[rand_file_index]), "c:/Users/tilli/OneDrive/Desktop/ImageClassificationProject/Training/" + str(classes[i]))
            c_files_temp.pop(rand_file_index)
        for k in range(len(c_files_temp)):
            shutil.copy(dataset + "\\" + str(classes[i]) + "\\"+ str(c_files_temp[k]), "c:/Users/tilli/OneDrive/Desktop/ImageClassificationProject/Testing/" + str(classes[i]))
            
            
def get_sample_filepaths(dataset_path, classes): # returns list of filepaths corresponding to data points given the overall path to the folder containing te dataset
    filepaths = []
    for i in range (len(classes)):
        class_files = os.listdir(dataset_path + "\\" + str(classes[i]))
        for j in range (len(class_files)):
            filepaths.append(dataset_path + "\\" + str(classes[i]) + "\\" + class_files[j])
    return filepaths
            
    

# create the training and testing datasets and save them to variables

# make_sets(dataset, classes)
train_path = "c:/Users/tilli/OneDrive/Desktop/ImageClassificationProject/Testing"
test_path = "c:/Users/tilli/OneDrive/Desktop/ImageClassificationProject/Testing"


train_dataset = bloodcells_dataset(get_sample_filepaths(train_path, classes))
test_dataset = bloodcells_dataset(get_sample_filepaths(test_path, classes))

# Dataloader to feed into neural network

batch_size = 4
train_loader = DataLoader(train_dataset, batch_size, shuffle = True) #tensor shape: [64, 3, 359, 359, 3]
test_loader = DataLoader(test_dataset, batch_size, shuffle = True)


         
            

#implement Convolutional Neural Network


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        
        self.conv1 = nn.Conv2d(3, 6, 15) # 3 rgb 15x15 kernel, output channels are 6
        
        
        
        
        self.pool = nn.MaxPool2d(3,3) #pool 2x2 pixels into one (stride = 2)
        self.pool2 = nn.MaxPool2d(5,5)
        self.pool3 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 15) 
        self.conv3 = nn.Conv2d(16, 20, 5)
        self.fc1 = nn.Linear(20 * 8 * 8, 120) #fully connected layer
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear (84, 8)
        
    def forward(self, x):
        # n, 3, 359, 359
        x = self.pool(F.relu(self.conv1(x)))  # output size of conv1 is 359-15 + 1 = 345 x 345 array with channel depth 6
        # output of first convolutional layer is n, 6, 345, 345
        # after pooling, (345-3)/3 + 1 = 114 x 114 pixel array with channel depth 6
        
        x = self.pool2(F.relu(self.conv2(x))) # output size of conv2 is 114 - 15 + 1 = 100 x 100 with channel depth 16
        # output of second convolutional layer is n, 16, 100, 100
        # after pooling, (100 - 5)/5 + 1 = 20 x 20 pixel array with channel depth 16
        
        x = self.pool3(F.relu(self.conv3(x))) # output size of conv3 is 20 - 5 + 1 = 16 x 16 array with channel depth 20
        # after pooling,  (16 - 2)/2 + 1 = 8 x 8 pixel array with channel depth 20
        
        # flatten 3d tensor to 1D tensor
        # print('x_shape:', x.shape)
        x = x.view(-1, 20 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



# write optimizer / learning loop




#hyper parameters

num_epochs = 16 #epoch refers to one complete pass of the training dataset through the algorithm
#batch_size = 16 # number of samples you feed into the model at each iteration of the training process (DEFINED ABOVE)
learning_rate = 0.01
model = ConvNet().to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
n_total_steps = len(train_loader)

total_loss = 0

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = torch.flatten(labels, start_dim = 1)
        labels = labels.to(device)
        # print('label shape:', labels.shape)

        # forward pass
        
        outputs = model(images)
        # print('outputs shape', outputs.shape)
        loss = criterion(outputs, labels)
        
        # backward and optimize
        optimizer.zero_grad()   # reset the gradient tensor to zero
        loss.backward()
        optimizer.step()
        
        
        if ((i + 1) % 100) == 0:
            print(f"Epoch[{epoch +1}/{num_epochs}], Step [{i+1 / n_total_steps}], Loss: {loss.item():.4f}")
            total_loss += float(loss.item())

print("Finished Training")
print(total_loss / (num_epochs * 8))


PATH = "./cell_set.pth"
torch.save(model.state_dict(), PATH) # save our neural network

def int_to_onehot(predictions: torch.Tensor) -> torch.Tensor:
    one_hot = []
    for i in range(predictions.size(dim = 0)):
        classification_array = [0 for i in range(8)]
        classification_array[predictions[i]] = 1
        one_hot.append(classification_array)
    return torch.tensor(one_hot)

def onehot_to_int(onehot_tensor: torch.Tensor) -> int:
    for i in range(8):
        if onehot_tensor[0][i] == 1:
            return i

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(8)]
    n_class_samples = [0 for i in range(8)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # maximize the returns
        _, predicted = torch.max(outputs, 1)
        n_samples += batch_size
        predicted = int_to_onehot(predicted).view(batch_size, 1, 8)
        predicted = predicted.to(device)
        #total_correct = (predicted == labels).sum().item()
        n_correct += (torch.sum((predicted == labels) * (predicted == 1))).item()
        #n_correct += ((predicted == labels).sum().item())/8
        # each value in predicted is an index 0 <= x <= 7 that describes the one-hot encoding   
        # create a new tensor that will store the one hot predictions (will eventually be 64, 1, 8)
        # for each value in predicted, create a new tensor and append to the main tensor created above
        # this will be our new predicted, converted to one hot.


        for i in range(batch_size):
            label = onehot_to_int(labels[i])
            pred = onehot_to_int(predicted[i])
            if label == pred:
                n_class_correct[label] += 1
            n_class_samples[label] += 1

        

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(8):
        class_acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {class_acc} %')
    






