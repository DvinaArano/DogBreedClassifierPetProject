import os
import pickle
from torchvision import datasets
import torchvision.transforms as transforms
import torch
from torchvision import models
import numpy as np
from PIL import ImageFile
from PIL import Image
import torch.nn as nn
import torch.optim as optim
ImageFile.LOAD_TRUNCATED_IMAGES = True

### TODO: Write data loaders for training, validation, and test sets
## Specify appropriate transforms, and batch_sizes


def load_obj():
    with open("Names.pkl", 'rb') as f:
        return pickle.load(f)


def load_input_image(img_path):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image = Image.open(img_path)
    prediction_transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                     transforms.ToTensor(),
                                     normalize])

    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = prediction_transform(image).unsqueeze(0)
    return image

def predict_breed_transfer(model, img_path):
    # load the image and return the predicted breed
    Names = load_obj()
    class_names = [item[4:].replace("_", " ") for item in Names['train'].dataset.classes]
    img = load_input_image(img_path)
    model = model.cpu()
    model.eval()
    result = model(img)
    resultnp = result.detach().cpu().numpy()
    i = 0
    output = ""
    while i < 4:
        idx = np.argmax(resultnp)
        value = round(np.amax(resultnp)*10)
        output = output + str(class_names[idx]) + ":" + str(value) +"%\n"
        np.put(resultnp,[idx],[0.0])
        i = i+1
    return output

def kickstart(img_path):

    model_transfer = models.resnet50(pretrained=True)

    for param in model_transfer.parameters():
        param.requires_grad = False

    model_transfer.fc = nn.Linear(2048, 133, bias=True)

    fc_parameters = model_transfer.fc.parameters()

    for param in fc_parameters:
        param.requires_grad = True

    model_transfer.load_state_dict(torch.load('./saved_models/model_transfer.pt', map_location='cpu'))

    results = predict_breed_transfer(model_transfer, img_path)

    return results


#for img_file in os.listdir('./images'):
#    img_path = os.path.join('./images', img_file)
#    predition = kickstart(img_path)
#    print("image_file_name: {0}, \t predition breed: {1}".format(img_path, predition))
