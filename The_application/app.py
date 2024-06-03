import streamlit as st

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Regression Data", "Graph Data"])

if page == "Regression Data":
    import regression_page
    regression_page.show()


elif page == "Graph Data":
    import graph_page
    graph_page.show()