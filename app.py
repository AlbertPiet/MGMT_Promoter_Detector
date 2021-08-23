import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics

import joblib 

from flask import Flask, jsonify, request, render_template,send_file


import os
import shutil
import sys 
import json
import glob
import random
import collections
import time
import re

import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import cv2


import torch
from torch import nn
from torch.utils import data as torch_data
from sklearn import model_selection as sk_model_selection
from torch.nn import functional as torch_functional
import torch.nn.functional as F

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

test=[]

mri_types = ['FLAIR','T1w','T1wCE','T2w']
#image size (images are square, and appear to be 512x512)
SIZE = 256
#Number of images to use (from center of scan)
NUM_IMAGES = 128
#Checkpoint file
Checkpoint_file="FLAIR-e10-loss0.685-auc0.572.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from efficientnet_pytorch_3d import EfficientNet3D
app = Flask(__name__)

#Function to return specific image
def load_dicom_image(path, img_size=SIZE, voi_lut=True):
    #load dicom image
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    #value of interest lookup table for images
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    data = cv2.resize(data, (img_size, img_size))
    return data

#Function to return array of images
def load_dicom_images_3d(num_imgs=NUM_IMAGES, img_size=SIZE, mri_type="FLAIR"):
#grabs all file paths for patient (scan_id) and mri type
    files = sorted(glob.glob(f"./data/*.dcm"), 
               key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
#getting num_images from the center of the mri (first and last images are often black, little information)
    middle = len(files)//2
    num_imgs2 = num_imgs//2
    p1 = max(0, middle - num_imgs2)
    p2 = min(len(files), middle + num_imgs2)
    #loading images in a list and then putting into an array with np.stack
    img3d = np.stack([load_dicom_image(f) for f in files[p1:p2]]).T 
    #creating blank images if there were not enough
    if img3d.shape[-1] < num_imgs:
        n_zero = np.zeros((img_size, img_size, num_imgs - img3d.shape[-1]))
        img3d = np.concatenate((img3d,  n_zero), axis = -1)
    #making image pixel values 0-1   
    if np.min(img3d) < np.max(img3d):
        img3d = img3d - np.min(img3d)
        img3d = img3d / np.max(img3d)     
    return np.expand_dims(img3d,0)

class Dataset(torch_data.Dataset):


    def __len__(self):
        return 1
    
    def __getitem__(self, index):
        data = load_dicom_images_3d()
        return {"X": torch.tensor(data).float(),'Id':0}
      
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = EfficientNet3D.from_name("efficientnet-b0", override_params={'num_classes': 2}, in_channels=1)
        n_features = self.net._fc.in_features
        self.net._fc = nn.Linear(in_features=n_features, out_features=1, bias=True)
    
    def forward(self, x):
        out = self.net(x)
        return out
 
def modelpred():
    current_directory = os.getcwd()
    data=os.path.join(current_directory, r'data')
    #os.mkdir(data)
    modelfile=Checkpoint_file
    print("Predict:", modelfile)
    data_retriever = Dataset(

    )
   
    data_loader = torch_data.DataLoader(
        data_retriever,
        batch_size=4,
        shuffle=False,
        num_workers=8,
    )
   
    model = Model()
    model.to(device)
    
    checkpoint = torch.load(modelfile,map_location=device )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    y_pred = []
    for e, batch in enumerate(data_loader,1):
        with torch.no_grad():
            tmp_pred = torch.sigmoid(model(batch["X"].to(device))).cpu().numpy().squeeze()
            if tmp_pred.size == 1:
                y_pred.append(tmp_pred)
            else:
                y_pred.extend(tmp_pred.tolist())
        
    shutil.rmtree(data)
    return y_pred   

@app.route('/')
def home():
    return render_template('predict.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    FileObj =  request.files['thefile']
    import zipfile
    with zipfile.ZipFile(FileObj, 'r') as zip_ref:
        zip_ref.extractall('data')
    #for m, mtype in zip(modelfiles,  mri_types):
    pred = modelpred()
    if(pred[0]<0.54):
        message="MGMT promoter is not present."
    else:
         message="MGMT promoter is present."
    return message+'The probability value is: '+str(pred[0])

if __name__ == "__main__":
    app.run(port='5000',host='0.0.0.0',debug=True)