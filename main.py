import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from PIL import Image

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.utils import to_categorical

"""
(X_train, y_train), (X_val, y_val) = cifar10.load_data()
X_train = X_train / 255
X_val = X_val / 255

y_train = to_categorical(y_train, 10)
y_val = to_categorical(y_val, 10)

model = Sequential([
    Flatten(input_shape=(32, 32, 3)),
    Dense(1000, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val, y_val))
"""
def main():
    st.title('Cifar10 Classifier')
    st.write('Upload image that fits into class and model will predict it')
    file = st.file_uploader('Upload image', type=['png', 'jpg', 'jpeg'])
    if file:
      image = Image.open(file)
      st.image(image, caption='Uploaded Image', use_column_width=True)

      resized_image = image.resize((32, 32))
      img_arr = np.array(resized_image / 255)
      img_arr = img_arr.reshape(1, 32, 32, 3)

      model = tf.keras.models.load_model('cifar10_model.h5')

      predictions = model.predict(img_arr)
      cifar10_class = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

      #DESIGN
      fig, ax = plt.subplots()
      y_pos = np.arange(len(cifar10_class))
      ax.barh(y_pos, predictions[0], align='center')
      ax.set_yticks(y_pos)
      ax.set_yticklabels(cifar10_class)
      ax.invert_yaxis()
      ax.set_xlabel('Probability')
      ax.set_title('Cifar10 Classifier')
      st.pyplot(fig)

    else:
        st.text('Please upload image')
