import streamlit as st
import pandas as pd
from sentiment_analysers.model import Model, FineGrainedModel, EmotionModel, \
    get_model, get_fine_grained_model, get_emotion_model
import requests
import re
import spacy_streamlit
import en_core_web_sm
nlp = en_core_web_sm.load()


model = get_model()
fine_grained_model = get_fine_grained_model()
emotion_model = get_emotion_model()

@st.cache()
def get_sentiment_analysis(df):
    sentiment_col = []
    fg_sentiment_col = []
    em_sentiment_col = []

    for result in df.body:
        sentiment, confidence, probabilities = model.predict(result)
        fg_sentiment, fg_confidence, fg_probabilities = fine_grained_model.predict(result)
        em_sentiment, em_confidence, em_probabilities = emotion_model.predict(result)
        print(sentiment, fg_sentiment, em_sentiment)
        if sentiment != "NEUTRAL":
            sentiment_col.append(sentiment)
            fg_sentiment_col.append(fg_sentiment)
            em_sentiment_col.append(em_sentiment)
        else:
            sentiment_col.append(sentiment)
            fg_sentiment_col.append(sentiment)
            em_sentiment_col.append(em_sentiment)

    # print(em_sentiment_col)
    # print(sentiment_col)
    # print(fg_sentiment_col)

    # df['sentiment'] = sentiment_col
    df['fg_sentiment'] = fg_sentiment_col
    df['em_sentiment'] = em_sentiment_col

    return df

def clean_text(text):
    # remove all unicode characters. In this case = emojis
    text = re.sub(r'[^\x00-\x7F]+', '', text) 
    return text

def get_search_result(search_query):

    url = "http://35.240.168.75:3000/search?search_term={}".format(search_query)
    payload={}
    headers = {}
    response = requests.request("GET", url, headers=headers, data=payload)
    search_result = response.json()

    search_result_dict = search_result['posts']
    search_result_df = pd.DataFrame.from_records(search_result_dict)

    return search_result_df


def main():
    st.title("Hello from sudo")
    st.write('\n')
    search_query = st.text_input('Enter your search query')
    search_query = clean_text(search_query)

    st.sidebar.markdown('**Filters**')
    search_result_number = st.sidebar.slider('number of search results', min_value=1, max_value=50)
    sentiment_filter = st.sidebar.multiselect('Sentiment of content', ['Positive', 'Negative', 'Neutral'])
    emotion_filter = st.sidebar.multiselect('Emotion of content', ['Happy', 'Sad', 'Angry', 'Confused'])

    search_bool = st.button('Search')
    advanced_search_bool = st.button('Advanced Search')

    if search_bool:

        search_result_df = get_search_result(search_query)

        temp_table = search_result_df[['body', 'creator', 'hashtags']]
        temp_table = temp_table[:search_result_number]

        st.table(temp_table)
    
    if advanced_search_bool:
    
        search_result_df = get_search_result(search_query)
        temp_table = search_result_df[['body', 'creator', 'hashtags']]
        temp_table = temp_table[:search_result_number]
        temp_table = get_sentiment_analysis(temp_table)
        st.table(temp_table)


    # st.markdown('__Sentiment__ of this review is: `{0}`.'.format(sentiment))
    # st.markdown('__Fine Grained Sentiment__ of this review is: `{0}`.'.format(fg_sentiment))
    # st.markdown('__Emotion__ of this review is: `{0}`.'.format(em_sentiment))


    # st.markdown('The confidence probabilities of the classes are `{0}`'.format(probabilities))
    # st.markdown('The confidence probabilities of the classes are `{0}`'.format(fg_probabilities))


    # st.write(search_query, search_result_number, sentiment_filter, emotion_filter)

if __name__ == "__main__":
    main()