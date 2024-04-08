import streamlit as st
import classify_page
import classify_video

st.set_page_config(
    page_title="Deforgify",
    page_icon="🤖",
    layout="wide")

PAGES = {
    "Classify Image": classify_page,
    "Classify Video": classify_video,
}


st.sidebar.title("Deforgify")

st.sidebar.write("Deforgify is a tool that utilizes the power of Deep Learning to distinguish Real images from the Fake ones.")

st.sidebar.subheader('Navigation:')
selection = st.sidebar.radio("", list(PAGES.keys()))

page = PAGES[selection]

page.app()