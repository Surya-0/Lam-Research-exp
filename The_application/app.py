import streamlit as st
from streamlit_lottie import st_lottie
import requests
from PIL import Image
import json

def load_lottie_file(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Load local Lottie animation
lottie_animation = load_lottie_file("Animation.json")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Regression Data", "Graph Data"])

if page == "Home":
    st.title("Welcome to the Capacity Planning App")
    st.write("This app helps you with capacity planning using neural networks and genetic algorithms.")

    if lottie_animation:
        st_lottie(lottie_animation, height=400, key="capacity-planning")

    # Adding image example
    image = Image.open("Capacity_Planning.png")
    st.image(image, caption="Capacity Planning", use_column_width=True)

elif page == "Regression Data":
    import regression_page
    regression_page.show()

elif page == "Graph Data":
    import graph_page
    graph_page.show()
