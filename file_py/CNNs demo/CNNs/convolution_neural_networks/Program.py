import numpy as np
import os
import sklearn as skl
import skimage as ski
import skimage.io as io
import cv2 as cv
import pickle


class Program(object):
    """description of class"""
    def __init__(self):
        self.data = (np.array([]), np.array([]));
        return;
    def read_data(self, path = 'D:\\data\\animals', trade = True):
        X = np.array([]);
        Y = np.array([]);
        if trade == False:
            subfolders = [f.path for f in os.scandir(path) if f.is_dir() ];
            count = 0;
            for i in subfolders:
                images = [];
                labels = [];
                for filename in os.listdir(i):
                    img = cv.imread(i + "\\" + filename);
                    if img is not None:
                        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY);
                        img = img/255.0;
                        images.append(img);
                    Y = np.append(Y, count);
                count+=1;
                X = np.append(X, images);
            pickle.dump(X, open('data.sav' , "wb"));
            pickle.dump(Y, open('label.sav', 'wb'));
        else:
            X = pickle.load(open("data.sav" , "rb"));
            #print(X.shape);
            Y = pickle.load(open("label.sav", "rb"));
            #print(Y.shape);
        self.data = (X, Y);
        return;
    def train_loop(self):

        return;
