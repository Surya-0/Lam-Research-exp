import streamlit as st
st.title("My first app")
st.header("header")
st.subheader('subheader')
st.write("Welcome")
st.text("Hi how are you")
st.markdown("""
# h1 tag
## h2 tag
### h3 tag
:moon:<br>
:sunglasses:
""",True)
d  = {1:'a',2:'b',3:'c',4:'d',5:'e'}
st.write(d)