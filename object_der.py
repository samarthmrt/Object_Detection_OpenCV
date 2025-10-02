import streamlit as st
import cv2
import numpy as np
import requests
import os



file=open('objects_list.txt')
li=file.read().split('\n')
classes=list(map(str.strip,li))
file.close()

# URLs for cfg and weights
cfg_url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg"
weights_url = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights"

if os.path.exists('yolov4.cfg') & os.path.exists('yolov4.weights'):
    pass
else:
    resp=requests.get(cfg_url)
    file=open('yolov4.cfg','wb')
    file.write(resp.content)
    file.close()

    resp=requests.get(weights_url)
    file=open('yolov4.weights','wb')
    file.write(resp.content)
    file.close()


model=cv2.dnn_DetectionModel('yolov4.cfg','yolov4.weights')
model.setInputSize(416,416)
model.setInputScale(1/255)

def detect(path):
    count_person=0
    file_bytes=np.asarray(bytearray(path.read()),dtype=np.uint8)
    img=cv2.imdecode(file_bytes,cv2.IMREAD_COLOR)
    classIds,classProbs,bboxes=model.detect(img,confThreshold=.65,nmsThreshold=.5)
    for box,cls,probs in zip(bboxes,classIds,classProbs):
        x,y,w,h=box
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.putText(img,f'{classes[cls]}({probs:.2f})',(x,y-10),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
    return img,count_person

# Set page config
st.set_page_config(page_title="Object Detection Pro", layout="wide")

# Custom CSS for shorter header
st.markdown(
    """
    <style>
    .full-header {
        height: 65vh; /* Reduced height */
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        background: linear-gradient(135deg, #1f4037, #99f2c8); /* Fancy gradient */
        color: white;
        font-family: 'Trebuchet MS', sans-serif;
        text-align: center;
        border-radius: 12px;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.4);
        margin-bottom: 20px;
    }
    .full-header h1 {
        font-size: 3.5rem;
        margin: 0;
        animation: fadeInDown 2s ease;
    }
    .full-header p {
        font-size: 1.3rem;
        margin-top: 10px;
        animation: fadeInUp 2s ease;
    }
    @keyframes fadeInDown {
        from {opacity: 0; transform: translateY(-50px);}
        to {opacity: 1; transform: translateY(0);}
    }
    @keyframes fadeInUp {
        from {opacity: 0; transform: translateY(50px);}
        to {opacity: 1; transform: translateY(0);}
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header content
st.markdown(
    """
    <div class="full-header">
        <h1>🚀 Object Detection</h1>
        <p>Object detection, Person count & Gender Detection ✨</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar styling with CSS
st.markdown(
    """
    <style>
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1f4037, #99f2c8);
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Sidebar headers */
    [data-testid="stSidebar"] h2 {
        color: #ffffff;
        text-align: center;
        font-size: 22px;
        border-bottom: 2px solid #ffffff33;
        padding-bottom: 5px;
    }

    /* Sidebar text */
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] div {
        color: #f0f0f0;
        font-size: 16px;
    }

    /* Contact box */
    .contact-box {
        background: rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 12px;
        margin-top: 15px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar Content
st.sidebar.image("Object Detection Pro Logo Design.png", use_container_width=True)

st.sidebar.header("👬 About Us")
st.sidebar.write("I am a *Machine Learning Engineer* specializing in *Object Detection*.")

st.sidebar.header("📞 Contact Us")
st.sidebar.markdown(
    """
    <div class="contact-box">
        📱 <b>7906877924</b><br>
        📧 samarthkumar0611@gmail.com
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.header("🌐 Connect With Me")
st.sidebar.markdown(
    """
    🔗 [LinkedIn](https://linkedin.com)  
    🐙 [GitHub](https://github.com)  
    """,
    unsafe_allow_html=True
)


# Main content
st.write("⬇ Scroll down for the rest of the app")
st.write("Here’s where your actual app content goes...")

uploaded_file=st.file_uploader('Choose a file',type=['png','jpeg','jfif'])

col1,col2=st.columns(2)
with col1:
    if uploaded_file:
        st.image(uploaded_file)
        btn=st.button('Prediction')
    
        with col2:
            if btn:
                img,count=detect(uploaded_file)
                st.image(img)
                