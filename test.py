import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from PIL import Image, ImageFile
import numpy as np
from matplotlib.legend_handler import HandlerLine2D
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
import torch.nn as nn

from torch.autograd import Variable
from PIL import Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

from classifier import Classifier

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

use_gpu = False
# use_gpu = True

transform = transforms.Compose(
        [transforms.Resize([128, 128]),
         transforms.RandomCrop(128),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])

classifier = Classifier(use_gpu=use_gpu)
classifier.load_state_dict(torch.load("food.fd"))


def predict_class(img):
    x = transform(img)
    x = x.reshape((1, 3, 128, 128))
    p = classifier.predict(x)
    cls = classes[torch.argmax(p)]
    return cls


def read_calories():
    cal = open('cal.txt', 'r')
    cals = {}
    for line in cal.readlines():
        splt = line.split()
        cals[splt[0]] = splt[-1]
    return cals


cals = read_calories()

# img = Image.open("sampleimages/macarons.jpg")
img = Image.open("sampleimages/donuts.jpg")
# img = Image.open("sampleimages/frozen_yoghurt.jpg")
# img = Image.open("sampleimages/steak.jpg")
# img = Image.open("sampleimages/baklava.jpg")
# img = Image.open("sampleimages/hot_dog.jpg")
# img = Image.open("sampleimages/ceasar_salad.jpg")
# img = Image.open("sampleimages/omelette.jpg")
cls = predict_class(img)
print("Predicted Food: %s \n%s kcal per 100g. " % (cls, cals[cls]))
