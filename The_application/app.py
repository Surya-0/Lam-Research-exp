import streamlit as st
from streamlit_lottie import st_lottie
import requests
from PIL import Image
import json

from bs4 import BeautifulSoup

def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        print(f"Error: Status code is {r.status_code}")
        print(f"Response text: {r.text}")
        return None
    soup = BeautifulSoup(r.text, 'html.parser')
    lottie_player = soup.find('dotlottie-player')
    if lottie_player is None:
        print("Error: Could not find dotlottie-player element in HTML")
        return None
    json_url = lottie_player['src']
    r = requests.get(json_url)
    if r.status_code != 200:
        print(f"Error: Status code is {r.status_code} for JSON URL")
        print(f"Response text: {r.text}")
        return None
    try:
        return r.json()
    except json.JSONDecodeError:
        print(f"Error: Response is not valid JSON for JSON URL")
        print(f"Response text: {r.text}")
        return None
# def load_lottie_file(filepath: str):
#     with open(filepath, "r") as f:
#         return json.load(f)
#
# # Load local Lottie animation
# lottie_animation = load_lottie_file("Animation.json")

# Load Lottie animation from a URL
lottie_url = "https://lottie.host/embed/17182c96-3ca9-417e-bbb3-cadabd7ba544/BWUY94Dvjj.json"
lottie_animation = load_lottie_url(lottie_url)
print(lottie_url)
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
