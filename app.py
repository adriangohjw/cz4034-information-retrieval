import streamlit as st
import pandas as pd
from sentiment_classifier.model import Model, get_model
from fine_grained_sentiment_classifier.model import FineGrainedModel, get_fine_grained_model
import re
import spacy_streamlit
import en_core_web_sm
nlp = en_core_web_sm.load()

def clean_text(text):
    # remove all unicode characters. In this case = emojis
    text = re.sub(r'[^\x00-\x7F]+', '', text) 
    return text

def main():
    st.title("Hello from sudo")
    st.write('\n')
    search_query = st.text_input('Enter your search query')
    search_query = clean_text(search_query)

    st.sidebar.markdown('**Filters**')
    search_result_number = st.sidebar.slider('number of search results', min_value=1, max_value=60)
    sentiment_filter = st.sidebar.multiselect('Sentiment of content', ['Positive', 'Negative', 'Neutral'])
    emotion_filter = st.sidebar.multiselect('Emotion of content', ['Happy', 'Sad', 'Angry', 'Confused'])
    model = get_model()
    fine_grained_model = get_fine_grained_model()

    sentiment, confidence, probabilities = model.predict(search_query)
    fg_sentiment, fg_confidence, fg_probabilities = fine_grained_model.predict(search_query)

    st.markdown('__Sentiment__ of this review is: `{0}`.'.format(sentiment))
    st.markdown('__Fine Grained Sentiment__ of this review is: `{0}`.'.format(fg_sentiment))

    # st.markdown('The confidence probabilities of the classes are `{0}`'.format(probabilities))
    # st.markdown('The confidence probabilities of the classes are `{0}`'.format(fg_probabilities))


    # st.write(search_query, search_result_number, sentiment_filter, emotion_filter)

if __name__ == "__main__":
    main()