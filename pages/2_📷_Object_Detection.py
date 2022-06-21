import torch
import streamlit as st
from PIL import Image
import time

st.set_page_config(
   page_title="Object Detection Demo",
   page_icon="📷",
   layout="wide",
   initial_sidebar_state="expanded",
)

@st.cache()
def load_yolov5():
    # ultralytics YOLOv5 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.hub.load(
        repo_or_dir='ultralytics/yolov5', 
        model='yolov5s',
        source='github'
    ).to(device).eval()

@st.cache(allow_output_mutation=True)
def inference(model, image, size=1280):
    return model(imgs, size=size) # includes Non-Maximum Suppression

if __name__ == '__main__':
    # Title
    st.title("📷 Object Detection Demo")
    st.info("Run a simple object detection model using Ultralytics YOLOv5. (Pre-trained on COCO)")
    
    # load model
    model = load_yolov5()
    assert model, "Failed to load model"

    with st.sidebar:
        st.header("Inference Hyperparameters")
        
        # Inference Hyperparameters
        model.conf = st.slider("NMS Threshold", min_value=0., max_value=1., value=0.25)
        model.iou = st.slider("IoU Threshold", min_value=0., max_value=1., value=0.5)
        # model.agnostic = st.select_slider("NMS Class-Agnostic", [False, True], value=False)
        # model.multi_label = st.select_slider("NMS multi-labels per box", [False, True], value=False)
        model.max_det = st.slider("Max Detections", min_value=None, max_value=1000, value=500)
        # model.amp = st.select_slider("Automatic Mixed Precision", [False, True], value=False)

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"], help='Image Only')
    
    if uploaded_file is not None:
        input_container, inference_container = st.columns(2)
        
        # First Column - Input Image
        with input_container:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(uploaded_file, caption='Input Image', use_column_width=True)
        
        # Second Column - Inference
        with inference_container:
            imgs = [image] # creating a batch
            start_time = time.time()
            results = inference(model, image)
            end_time = time.time()
            st.image(results.render()[0], caption='Output Image', use_column_width=True)
            time_taken = end_time - start_time
        
        with st.expander("Results"):
            st.metric(label='Inference Time', value="{:.3f} img/s".format(time_taken))
            st.dataframe(results.pandas().xyxy[0])