import streamlit as st
import pandas as pd

def main():
    input_short_txt = st.text_input("Short Text Input")
    input_long_txt = st.text_area("Long Text Input")
    input_image = st.file_uploader("Upload images", type=["jpg", "png"])

    if input_image:
        st.image(input_image)

if __name__ == "__main__":
    main()