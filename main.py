import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st


def calculate_similarity(user_input):
    df = pd.read_csv('elonmusk.csv')

    df = df[['Text']]

    vectorizer = CountVectorizer()

    X = vectorizer.fit_transform(df['Text'])

    user_vector = vectorizer.transform([user_input])

    similarity_scores = cosine_similarity(X, user_vector)

    most_similar_tweet_index = similarity_scores.argmax()

    most_similar_tweet = df.loc[most_similar_tweet_index, 'Text']

    user_words = vectorizer.inverse_transform(user_vector)[0]
    tweet_words = vectorizer.inverse_transform(X[most_similar_tweet_index])[0]

    total_unique_words_user = len(set(user_words))
    total_unique_words_tweet = len(set(tweet_words))
    matched_words = len(set(user_words).intersection(tweet_words))
    percentage_matched = (matched_words / total_unique_words_tweet) * 100 if total_unique_words_tweet != 0 else 0

    # Return the result
    return most_similar_tweet, percentage_matched

st.title('Do You Sound Like Elon?')


with st.container():
    user_input = st.text_area('', height=300, placeholder='Write Tweet Here', max_chars=280, )

    if st.button('Analyze tweet'):
        most_similar_tweet, percentage_matched = calculate_similarity(user_input)
        st.text("Your sentence closely resembles Elon Musk's tweet: \n{}".format(most_similar_tweet))
        st.subheader("You sound {:.2f}%".format(percentage_matched) + " like Elon")
