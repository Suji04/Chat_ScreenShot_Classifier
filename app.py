import base64
import numpy as np
import io
from PIL import Image
import keras
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

classes = ["CHAT", "NOT A CHAT"]

def get_model():
    global model
    model = load_model("model.h5")
    model._make_predict_function()
    print("Model Loaded")
    
def preprocess_img(image, target_size, inv):
    # image = image.convert("RGB")
    image = image.resize(target_size)
    if inv==True :
        image=np.invert(image)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.
    return image    

print("loading model...")
get_model()


@app.route('/')
def index():
    return render_template("index.html")
    

@app.route("/predict-image/", methods = ["GET","POST"])
def predict_img():
    message = request.get_json(force=True)
    encoded = message["image"]
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_img = preprocess_img(image, target_size=(64,64), inv=False)
    pred = model.predict(processed_img)
    print(pred)
    idx = 0
    if pred>0.5: 
        idx=1
    print(idx)
    response = {
            'predictionImg' : str(classes[idx])
    }
    return jsonify(response)
