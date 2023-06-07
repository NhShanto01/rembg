import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import uuid
import os

from model import U2NET
from torch.autograd import Variable
from skimage import io, transform
from PIL import Image


from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil
import uvicorn

app = FastAPI()







# Get The Current Directory
currentDir = os.path.dirname(__file__)

# Functions:
# Save Results


def save_output(image_name, output_name, pred, d_dir, type):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    im = Image.fromarray(predict_np*255).convert('RGB')
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]))
    pb_np = np.array(imo)
    if type == 'image':
        # Make and apply mask
        mask = pb_np[:, :, 0]
        mask = np.expand_dims(mask, axis=2)
        imo = np.concatenate((image, mask), axis=2)
        imo = Image.fromarray(imo, 'RGBA')

    imo.save(d_dir+output_name)
# Remove Background From Image (Generate Mask, and Final Results)


@app.post("/removeBG/")
async def removeBG(image_file: UploadFile):

    # ------- Load Trained Model --------
    print("---Loading Model---")
    model_name = 'u2net'
    model_dir = os.path.join(currentDir, 'saved_models',
                             model_name, model_name + '.pth')
    net = U2NET(3, 1)
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    # ------- Load Trained Model --------

    inputs_dir = os.path.join(currentDir, 'static/inputs/')
    results_dir = os.path.join(currentDir, 'static/results/')
    masks_dir = os.path.join(currentDir, 'static/masks/')

    # Save image to inputs folder
    unique_filename = str(uuid.uuid4()) + '.jpg'
    image_path = os.path.join(inputs_dir, unique_filename)
    with open(image_path, 'wb') as output_file:
        shutil.copyfileobj(image_file.file, output_file)
         # Process the image
    img = cv2.imread(image_path)
    image = transform.resize(img, (320, 320), mode='constant')

    tmpImg = np.zeros((image.shape[0], image.shape[1], 3))

    tmpImg[:, :, 0] = (image[:, :, 0]-0.485)/0.229
    tmpImg[:, :, 1] = (image[:, :, 1]-0.456)/0.224
    tmpImg[:, :, 2] = (image[:, :, 2]-0.406)/0.225

    tmpImg = tmpImg.transpose((2, 0, 1))
    tmpImg = np.expand_dims(tmpImg, 0)
    image = torch.from_numpy(tmpImg)

    image = image.type(torch.FloatTensor)
    image = Variable(image)
    d1, d2, d3, d4, d5, d6, d7 = net(image)
    pred = d1[:, 0, :, :]
    ma = torch.max(pred)
    mi = torch.min(pred)
    dn = (pred-mi)/(ma-mi)
    pred = dn

    # Save results to results folder
    save_output(image_path, unique_filename[:-4] + '.png', pred, results_dir, 'image')

    # Save mask to masks folder
    save_output(image_path, unique_filename[:-4] + '.png', pred, masks_dir, 'mask')

    return {"result": "Success"}

print("---Removing Background...")
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)