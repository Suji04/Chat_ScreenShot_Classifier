#!/usr/bin/env python
"""
Generate predictions and show scores for screenshot filenames listed on command line.

TODO:
 Use labels from training directores, don't assume a classification name
 Provide option to show images (show=True).
"""

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import sys
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.image import image
import numpy as np
import glob

classifier=load_model("model.h5")

def load_image(img_path, show=True):
    img_original = image.load_img(img_path)
    img = image.load_img(img_path, target_size=(48, 54))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]
    if show:
        plt.imshow(img_original)
        plt.axis('off')
        plt.show()
    return img_tensor

for img_file in sys.argv[1:]:
  new_image = load_image(img_file, show=False)
  pred = classifier.predict(new_image)[0][0]
  print(f'score={pred:.6f} {" map " if pred < 0.5 else "other"} {img_file}')
