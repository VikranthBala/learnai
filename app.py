import streamlit as st

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

mnist = fetch_openml('mnist_784',as_frame=False,cache=True)

print(mnist.data.shape)

st.write("hi but this is confusing as shit")


