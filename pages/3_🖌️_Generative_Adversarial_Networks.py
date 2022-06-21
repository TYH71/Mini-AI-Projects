import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import keras
import matplotlib.pyplot as plt

st.set_page_config(
   page_title="Demo",
   page_icon="🖌️",
   layout="wide",
   initial_sidebar_state="expanded",
)

# class names for CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

@st.cache(allow_output_mutation=True)
def load_generator(path:str='pages/assets/CIFAR_GAN/CGAN_Generator.h5'):
   return load_model(path)

@st.cache(hash_funcs={keras.engine.functional.Functional: lambda _:None})
def generate_image(generator, condition):
   latent_z = np.random.normal(size=(100, 128))
   condition = np.full(100, condition, dtype=np.int32)
   condition = tf.keras.utils.to_categorical(condition, num_classes=10)
   return generator.predict([latent_z, condition])

if __name__ == "__main__":
   st.write("# 🖌️ Generative Adversarial Networks")
   st.info("Generate CIFAR-10 images using Auxiliary Classifier GAN (AC-GAN)")
   
   gan_generator = load_generator()

   with st.sidebar:
      # st.header("Generator Hyperparameters")
      condition = st.selectbox("Condition", np.arange(0, 10), index=0, format_func=lambda x: class_names[x])
   
   with st.container():
      imgs = generate_image(gan_generator, condition)
   
      # plotting
      fig = plt.figure(figsize=(20, 20), tight_layout=True)
      for idx, img in enumerate(imgs):
         ax = fig.add_subplot(10, 10, idx+1)
         ax.imshow((img+1)/2)
         ax.axis('off')
      fig.suptitle(class_names[condition], y=1, fontsize=28)
      st.pyplot(fig)

