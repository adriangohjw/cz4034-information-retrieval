import streamlit as st
import pandas as pd
from sentiment_analysers.model import Model, FineGrainedModel, EmotionModel, \
    get_model, get_fine_grained_model, get_emotion_model
import requests
import re
import copy
from json_flatten import flatten_json
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

    for result in df._source_body:
        result = clean_text(result)
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

    df['fg_sentiment'] = fg_sentiment_col
    df['em_sentiment'] = em_sentiment_col

    return df

def clean_text(text):
    # remove all unicode characters. In this case = emojis
    text = re.sub(r'[^\x00-\x7F]+', '', text) 
    return text

def get_search_result(search_query, search_result_number):

    url = "http://35.240.168.75:3000/search?search_term={}&size={}".format(search_query, search_result_number)
    payload={}
    headers = {}
    response = requests.request("GET", url, headers=headers, data=payload)
    search_result = response.json()
    search_results_list = []

    for i in search_result['posts']:
        search_results_list.append(flatten_json(i))

    search_result_df = pd.DataFrame.from_records(search_results_list)

    return search_result_df

def get_search_suggestion(search_query):

    url = "http://35.240.168.75:3000/suggest?search_term={}".format(search_query)

    payload={}
    headers = {}
    response = requests.request("GET", url, headers=headers, data=payload)
    suggest_results = response.json()

    suggest_results_df = pd.DataFrame.from_records(suggest_results['posts'])
    
    return suggest_results_df

def get_search_histogram(search_query, metric, search_result_number):

    url = "http://35.240.168.75:3000/distribution?search_term={}&field={}&size={}".format(search_query, metric, search_result_number)
    payload={}
    headers = {}
    response = requests.request("GET", url, headers=headers, data=payload)
    metric_distribution_dict = response.json()['distribution']

    metric_distribution = pd.Series(metric_distribution_dict, name='value')
    metric_distribution.index.name = 'percentile'
    metric_distribution = metric_distribution.reset_index()
    # metric_distribution = metric_distribution.set_index('percentile')

    print(metric_distribution)

    return metric_distribution
    
def main():
    st.markdown("# :wave: Hello! Welcome to `search.io`")
    st.markdown("#### A blazing fast :rocket: search engine for Parler data")
    html_string = "<br>"
    st.markdown(html_string, unsafe_allow_html=True)
    string ='''
    
            We have indexed **85 Million** posts from Parler before it was taken down. You can :mag: find the posts using the search term or even using certain :chart_with_upwards_trend: trending hashtags 
            '''
    st.markdown(string)
    search_query = st.text_input('Enter your search query')
    search_query = clean_text(search_query)

    st.sidebar.markdown('**Search Help**')
    st.sidebar.markdown('`Search` returns the users and posts ranked')
    st.sidebar.markdown('`Advanced Search` returns the normal search result with Sentiment & Emotion Analysis running in real-time')
    st.sidebar.markdown('`Suggest Search term` returns related search queries for the input search if there are any available')

    st.sidebar.markdown(html_string, unsafe_allow_html=True)
    
    st.sidebar.markdown('**Search Filters**')
    search_result_number = st.sidebar.slider('Number of search results', min_value=1, max_value=50)

    st.sidebar.markdown(html_string, unsafe_allow_html=True)

    st.sidebar.markdown('**Advanced Search filters**')
    sentiment_filter = st.sidebar.multiselect('Sentiment of content', ["SOMEWHAT NEGATIVE", "SOMEWHAT POSITIVE", "VERY NEGATIVE", "VERY POSITIVE", "NEUTRAL"])
    emotion_filter = st.sidebar.multiselect('Emotion of content', ['HAPPY', 'SAD', 'ANGRY', 'DISGUST', 'SURPRISE', 'FEAR'])

    search_bool = st.button('Search')
    advanced_search_bool = st.button('Advanced Search')
    suggest_bool = st.button('Suggest Search term')

    if search_bool:

        search_result_df = get_search_result(search_query, search_result_number)

        # temp_table = search_result_df[['body', 'creator', 'hashtags', 'reach_score']]
        temp_table = search_result_df[['_source_body', '_source_username']]



        # temp_table = temp_table[:search_result_number]
        
        st.markdown('### Search Results')
        st.table(temp_table)
        
        st.markdown(html_string, unsafe_allow_html=True)

        st.markdown('### Result Distribution Plots')

        st.markdown(html_string, unsafe_allow_html=True)

        # st.markdown('** Upvotes distribution in various percentiles **')
        # upvote_df = get_search_histogram(search_query, 'upvotes', search_result_number)
        # st.line_chart(upvote_df['value'])
        # st.markdown(html_string, unsafe_allow_html=True)

        # st.markdown('** Followers distribution in various percentiles **')
        # followers_df = get_search_histogram(search_query, 'followers', search_result_number)
        # st.line_chart(followers_df['value'])
        # st.markdown(html_string, unsafe_allow_html=True)

        # st.markdown('** Impressions distribution in various percentiles **')
        # impressions_df = get_search_histogram(search_query, 'impressions', search_result_number)
        # st.line_chart(impressions_df['value'])
        # st.markdown(html_string, unsafe_allow_html=True)

        metrics = ['_source_upvotes','_source_followers', '_source_impressions']

        for i in metrics: 
            st.line_chart(search_result_df[i])
    
    if advanced_search_bool:
    
        search_result_df = get_search_result(search_query, search_result_number)
        mod_table = search_result_df[['_source_body', '_source_username', '_score']]
        mod_table = get_sentiment_analysis(mod_table)
        temp_table = copy.deepcopy(mod_table)

        print(emotion_filter, sentiment_filter)

        if sentiment_filter:
            temp_table = temp_table.loc[temp_table['fg_sentiment'].isin(sentiment_filter)]
            print("sf")
            print(temp_table)

        if emotion_filter:
            temp_table = temp_table.loc[temp_table['em_sentiment'].isin(emotion_filter)]
            print("sf")
            print(temp_table)

        st.table(temp_table)

    if suggest_bool:

        suggest_results_df = get_search_suggestion(search_query)
        st.table(suggest_results_df)

if __name__ == "__main__":
    main()