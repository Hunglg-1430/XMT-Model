import streamlit as st
import image_prediction
from PIL import Image

def app():
    # st.image('./Streamlit_UI/Header.gif', use_column_width=True)
    st.write("Upload a Picture to see if it is a fake or real face.")
    file_uploaded = st.file_uploader("Choose the Image File", type=["jpg", "png", "jpeg"])
    if file_uploaded is not None:
        processed_image = image_prediction.process_and_save_image(file_uploaded)
        c1, buff, c2 = st.columns([2, 0.5, 2])
        c2.subheader("Classification Result")
        st.image(processed_image,  width=600)



