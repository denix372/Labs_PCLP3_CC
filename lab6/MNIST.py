import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import gradio as gr
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from PIL import Image, ImageOps


def predict_digit(img):
    img_flat = img.reshape(1, 784).astype('float32')
    pred = model.predict(img_flat)[0]
    return int(pred)

def center_and_resize(img):
    img = ImageOps.invert(img)
    bbox = img.getbbox()
    img = img.crop(bbox)
    img = ImageOps.invert(img)

    img = img.resize((20, 20), Image.Resampling.LANCZOS)
    new_img = Image.new("L", (28,28), 0)
    new_img.paste(img, (4, 4))
    return new_img

def predict_digit(img):
    try:
        img = img["composite"]
        img = Image.fromarray(img[..., :3]).convert("L")
        img = ImageOps.invert(img)
        img = center_and_resize(img)

        img_np = np.array(img)
        img_flat = img_np.reshape(1, 784) /255.0

        pred_class = model.predict(img_flat)[0]
        proba = model.predict_proba(img_flat)[0]

        return f"Predictie: {pred_class}"
    
    except Exception as e:
        return f"Eroare: {str(e)}"

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(np.int32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=60000, test_size=10000, random_state=42, shuffle=True
)

labels, counts = np.unique(y_train, return_counts=True)
freq_series = pd.Series(counts, index=labels)

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)


cm = confusion_matrix(y_test, y_pred)

labels = [str(i) for i in range(10)]

interface = gr.Interface(
    fn = predict_digit,
    inputs = gr.Sketchpad(),
    outputs = "text",
    title = "Recunoastere cifre scrise de mana",
    description= "Deseneaza o cifra si apasa Submit"
)

interface.launch()