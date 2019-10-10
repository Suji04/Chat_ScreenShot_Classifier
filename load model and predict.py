import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.image import image
import numpy as np
import glob

classifier=load_model("model.h5")

def load_image(img_path, show=True):
    img_original = image.load_img(img_path)
    img = image.load_img(img_path, target_size=(64, 64))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]
    if show:
        plt.imshow(img_original)                           
        plt.axis('off')
        plt.show()
    return img_tensor

'''for img in glob.glob("test*.jpg"):
    new_image = load_image(img)
    pred = classifier.predict(new_image)
    if pred<.5 : print("chat")
    else : print("not chat")'''

new_image = load_image("test.png")
pred = classifier._make_predict_function(new_image)
if pred<.5 : print("chat")
else : print("not chat")

