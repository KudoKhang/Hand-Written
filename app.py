import numpy as np
import gradio as gr
import cv2

# Ignore warnings in output
import warnings
warnings.filterwarnings("ignore")

from tensorflow.keras.models import load_model

word_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}
model = load_model('modelHandWritten.h5')

def classify(img):
    img_final = cv2.resize(img, (28, 28))
    img_final = np.reshape(img_final, (1, 28, 28, 1))
    prediction = model.predict(img_final).flatten()
    return {word_dict[i]: float(prediction[i]) for i in range(25)}


    # img_pred = word_dict[np.argmax(list(model.predict(img_final)[0]))]
    # return img_pred

iface = gr.Interface(
    classify, 
    gr.inputs.Image(shape=(224, 224), image_mode='L', invert_colors=True, source="canvas"),
    gr.outputs.Label(num_top_classes=3),
    capture_session=True,
    )

if __name__ == "__main__":
    iface.launch(share=True)








# FOR IMG INPUT

# def classify(img):
    # img = cv2.GaussianBlur(img, (7, 7), 0)
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # _, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)

    # img_final = cv2.resize(img_thresh, (28, 28))
    # img_final = np.reshape(img_final, (1, 28, 28, 1))
    # img_pred = word_dict[np.argmax(list(model.predict(img_final)[0]))]
    # return img_pred

# iface = gr.Interface(
    # classify, 
    # gr.inputs.Image(shape=(224, 224)),
    # gr.outputs.Label(),
    # capture_session=True,
    # examples=[
    #     ["./imgTest/b.png"]
    # ]
    # )

# img_pred = word_dict[np.argmax(list(model.predict(img_final)[0]))]
# <=>
# result = model.predict(img_final)[0]
# l = list(result) 
## l = model.predict(img_final).toList()[0]
# print(word_dict[np.argmax(l)])
