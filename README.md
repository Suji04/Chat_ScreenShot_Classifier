# Chat_ScreenShot_Classifier
This convolutional neural network can predict if an input image is a chat screen-shot or a normal image. I have trained the model on 527 training images and tested on 132 test images. After 5 epochs it reached a training set accuracy more than 99% and a test set accuracy more than 98%. 

# Motivation : 
I have a couple of friends who used to send me a lot of chat screen-shots. Gradually my phone gallery became filled with them. So I thought it would be nice if I can make a model which can separate the chat screen-shots from other images for me :P 

# Try yourself :
https://screen-shot-classifier.herokuapp.com/

# Model :
Download the model from : https://drive.google.com/open?id=1k99ndVPuxI3kDGs6or2rSUWPC5LSjF-9

# Try standalone prediction code.
python screenshot_predict.py [files]

# To train a new model, first populate files in two subdirectories each of directories named test_set and training_set,
# You can pick random files for testing and move them to the test directory.  Specify the number to move as 10-20% of samples
cd training_set/chats
ls | shuf -n 20 | xargs -i mv {} ../../test_set/chats
cd ../others
ls | shuf -n 20 | xargs -i mv {} ../../test_set/others
cd ../..

# Then run the training code.
# It may be necessart to adjust `steps_per_epoch` in the code to match #images / batch_size
python screenshot_train.py
