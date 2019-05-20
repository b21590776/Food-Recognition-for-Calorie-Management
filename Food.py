import os
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from PIL import Image, ImageFile
import numpy as np
from project import train_on_gpu
from matplotlib.legend_handler import HandlerLine2D
import matplotlib.pyplot as plt
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_data_path = r'C:\Users\Berke\Desktop\food-101\train'
test_data_path = r'C:\Users\Berke\Desktop\food-101\test'
# Hyper parameters
num_epochs = 15
num_classes = 101
batch_size = 64
learning_rate = 0.001

import time
start = time.time()

def load_training(train_data_path, batch_size):
    transform = transforms.Compose(
        [transforms.Resize([128,128]),
         transforms.RandomCrop(128),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    data = datasets.ImageFolder(root=train_data_path, transform=transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    return train_loader

def load_test(test_data_path, batch_size):
    transform = transforms.Compose(
        [transforms.Resize([128,128]),
         transforms.RandomCrop(128),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    data = datasets.ImageFolder(root=test_data_path, transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    return test_loader



# Data loader
train_load = load_training(train_data_path, batch_size)

test_load = load_test(test_data_path, batch_size)

# Convolutional neural network (two convolutional layers)
class Classifier(nn.Module):
    use_gpu = True
    def __init__(self, use_gpu=True):
        super(Classifier, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)).cuda()
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)).cuda()
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)).cuda()
        self.fc1 = nn.Linear(32768, num_classes).cuda()


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        return out

    def predict(self, x):
        return self.forward(x)

import numpy as np

model = Classifier(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
def train(epoch, model, train_loader, optimizer):
    correct_train = 0
    train_loss = 0.0
    temp = 0.0
    # model is setting to train
    for batch_i, (data, target) in enumerate(train_load):
        data = data.cuda()
        labels = target.cuda()

        # moving tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # clearing the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: computing predicted outputs by passing inputs to the model
        output = model(data)
        # calculating the batch loss
        loss = criterion(output, target)
        # backward pass: computing gradient of the loss with respect to model parameters
        loss.backward()
        # performing a single optimization step (parameter update)
        optimizer.step()
        # updating training loss
        train_loss += loss.item()
        temp += loss.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct_train += pred.eq(labels.data.view_as(pred)).cpu().sum()
        loss = temp / batch_size

        # printing training loss every specified number of mini-batches
        if batch_i % batch_size == batch_size - 1:
            print('Epoch %d, Batch %d, Loss: %.7f' % (epoch, batch_i + 1, loss))
            temp = 0.0

    train_loss /= len(train_load)
    print('\nTrain set = Epoch: {}\tAccuracy {}/{} ({:.0f}%)\tAverage loss: {:.7f}\n'.format(
        epoch, correct_train, len(train_load) * batch_size,
                                  100. * correct_train / (len(train_load) * batch_size), train_loss))

    acc = '{:.0f}'.format(100. * correct_train / (len(train_load) * batch_size))
    return train_loss, float(acc) / 100


x = list()
train_y = list()
train_a = list()
for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train(epoch, model, train_load, optimizer)
    x.append(epoch)
    train_a.append(train_acc)
    train_y.append(train_loss)

# class names
classes = np.array(['apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
           'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito',
           'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake', 'ceviche',
           'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla', 'chicken_wings',
           'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder', 'club_sandwich', 'crab_cakes',
           'creme_brulee', 'croque_madame', 'cup_cakes', 'deviled_eggs', 'donuts', 'dumplings', 'edamame',
           'eggs_benedict', 'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras', 'french_fries',
           'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice', 'frozen_yogurt', 'garlic_bread',
           'gnocchi', 'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon', 'guacamole', 'gyoza', 'hamburger',
           'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna', 'lobster_bisque',
           'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup', 'mussels', 'nachos', 'omelette',
           'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck', 'pho', 'pizza',
           'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake',
           'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese',
           'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake', 'sushi', 'tacos', 'takoyaki',
           'tiramisu', 'tuna_tartare', 'waffles'])

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
# confusion matrix function ref: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix
# .htmlsphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_load:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


    print('Overall test Accuracy : {} %'.format(100 * correct / total))

    # np.set_printoptions(precision=2)

    # # Plot non-normalized confusion matrix
    # plot_confusion_matrix(labels, predicted, classes=classes,
    #                       title='Confusion matrix, without normalization')
    #
    # # Plot normalized confusion matrix
    # plot_confusion_matrix(labels, predicted, classes=classes, normalize=True,
    #                       title='Normalized confusion matrix')
    #
    # plt.show()
print("Program is finished in" + str(time.time() - start) + " seconds.")


def plot_accuracy_table(x, train_y,str):
    fig = plt.figure(0)
    fig.canvas.set_window_title(str+'accuracy')
    plt.axis([0, num_epochs + 1, 0, 1])
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    train_graph, = plt.plot(x, train_y, 'b--', label='Train accuracy')
    plt.legend(handler_map={train_graph: HandlerLine2D(numpoints=3)})
    plt.show()


def plot_loss_table(x, train_y,str):

    fig = plt.figure(0)
    fig.canvas.set_window_title(str+' loss')
    plt.axis([0, num_epochs + 1, 0, 5.5])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    train_graph, = plt.plot(x, train_y, 'b--', label='Train loss')
    plt.legend(handler_map={train_graph: HandlerLine2D(numpoints=3)})
    plt.show()

plot_loss_table(x, train_y, "train")
plot_accuracy_table(x, train_a, "train")

# Save the model checkpoint
torch.save(model.state_dict(), 'food.fd')
