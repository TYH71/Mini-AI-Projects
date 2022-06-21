import streamlit as st
import numpy as np
# from skimage import io, color, img_as_ubyte
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
# import matplotlib.pyplot as plt

st.set_page_config(
   page_title="Color Quantization",
   page_icon="🎨",
   layout="wide",
   initial_sidebar_state="expanded",
)

@st.cache()
def predict(img_data, clusters: int):
    return MiniBatchKMeans(
        n_clusters=clusters,
        max_iter=500,
        batch_size=3072,
        tol=0.01
    ).fit(img_data, clusters)
    

if __name__ == "__main__":
    # Title
    with st.container():
        st.write("# 🎨 Color Quantization")
        st.info("Conduct Color-Based Segmentation of an Image based on the chromaticity plane leveraging over an unsupervised learning technique, K-Means Clustering, to reduce the number of distinct colours.")
    
    # Explanation
    with st.expander('Discussion'):
        st.markdown('''
        K-Means Clustering is an iterative clustering algorithm that attempts to find the centroids based on the input number of clusters (denoted as $k$) given. 
        The procedure for K-Means is as follows:

        1.	Initialization: initiate random centroids based on $k$
        2.	Assignment: assign to the nearest centroid
        3.	Update: update the centroid with a new cluster
        4.	Repeat steps 2 and 3 until convergence

        The idea is to locate the ($r_n$, $g_n$, $b_n$) at the centroids which is done using the Euclidean Distance or “straight-line distance”.
        By using K-Means, the algorithm could effectively locate and replace the cluster region with the centroid RGB values.
        ''')
        st.latex(r"Distance = \sqrt{(r_1 - r_2)^2+(g_1-g_2)^2+(b_1-b_2)^2}")
        st.code('''
from sklearn.cluster import MiniBatchKMeans

# initiate class and fit the image data
model = MiniBatchKMeans(n_clusters=4, batch_size=3072)
model.fit(image_data)

# update every pixel with the assigned color/centroid
k_colors = model.cluster_centers_[model.labels_]        
        ''', language="python")
    
    with st.sidebar:
        cluster_parameters = st.slider("Number of Clusters", min_value=2, max_value=256, value=64)
    
    # Load Image
    buffer_file = st.file_uploader("Upload an image", type=["jpg", "png"], accept_multiple_files=False, help='Image Only')
    
    if buffer_file is not None:
        img = Image.open(buffer_file).convert("RGB")
        img_data = np.asarray(img)
        img_preprocess = (img_data/255.0).reshape(-1, 3)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(img, caption='Input Image', use_column_width=True)
        
        with col2:
            # Generating Image
            km = predict(img_preprocess, cluster_parameters)
            
            # Generating Quantized Image
            k_colors = km.cluster_centers_[km.labels_]
            k_img = np.reshape(k_colors, (img_data.shape[0], img_data.shape[1], 3))
            st.image(k_img, caption='Quantized Image', use_column_width=True)