import streamlit as st
import pandas as pd


def main():
    st.title("Hello from sudo")
    st.write('\n')
    search_query = st.text_input('Enter your search query')

    st.sidebar.markdown('**Filters**')
    search_result_number = st.sidebar.slider('number of search results', min_value=1, max_value=60)
    sentiment_filter = st.sidebar.multiselect('Sentiment of content', ['Positive', 'Negative', 'Neutral'])
    emotion_filter = st.sidebar.multiselect('Emotion of content', ['Happy', 'Sad', 'Angry', 'Confused'])

            
    st.write(search_query, search_result_number, sentiment_filter, emotion_filter)

if __name__ == "__main__":
    main()