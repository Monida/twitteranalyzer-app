from flask import Flask, render_template, request, redirect
import pandas as pd
from twitter_analyzer.Twitter import Twitter
import os
import base64
from rq import Queue
from worker import conn
from redis import Redis
import time

q = Queue('low', connection=conn)

app = Flask(__name__)
app.vars = {}
twitter = Twitter('')


@app.route('/', methods=["GET", "POST"])
def enter_query():
    if request.method == "GET":
        return render_template('enterquery.html')
    else:
        clean_search()
        return render_template('enterquery.html')

@app.route('/analyzing', methods=["GET","POST"])
def download_tweets():
    if request.method == "POST":
        # Coming from enterquery.html
        if request.form['query'] == '':
            return redirect('/')
            
        else:

            twitter.query = request.form['query']

            # Send get_tweets to a background worker
            job=q.enqueue(twitter.get_tweets,twitter.query,job_id='get_tweets_job')

            return render_template('analyzing.html')



@app.route('/show_analysis', methods=["GET", "POST"])
def return_query():
    if request.method == "GET":
        # Coming from moreinsights.html

        return render_template('returnquery.html', query=twitter.query, num_of_tweets=app.vars['num_of_tweets'],
                               table=twitter.top_words['words'].to_html(index=False, header=False))

    if request.method == "POST":

        # Coming from analyzing.html

        job = q.fetch_job('get_tweets_job')

        if job.result == None:
            return render_template('analyzing.html')

        else:
            # Send get_tweets task to background worker

            data=job.result

            # Reformat tweets
            
            tweets = twitter.reformat_tweets(data)

            # Clean tweets
            
            tweets = twitter.clean_and_tokenize()
            app.vars['num_of_tweets'] = len(tweets)

            # Perform topic modeling
            
            tweets = twitter.manualModelling()
            vectorized_tweets = twitter.vectorize_tweets()
            matrix = vectorized_tweets['words_matrix']
            feature_names = vectorized_tweets['feature_names']
            lda_model = twitter.fit_LDA(matrix)

            twitter.LDA_top_words(lda_model, feature_names)

            # Plot clustergram

            twitter.top_labeled_topics()
            clusters_data = twitter.cluster_text()
            twitter.create_clustergram(twitter.topics)
            topic_table = twitter.top_words['words']

            return render_template('returnquery.html', query=twitter.query,
                                   num_of_tweets=app.vars['num_of_tweets'],
                                   table=topic_table.to_html(index=False, header=False),
                                   clustergram=twitter.clustergram)



    

 #if tweets.empty:
 #   return redirect('/error')

@app.route('/moreinsights', methods=["POST"])
def more_insights():
# Check what button was clicked on returnquery.html

    if request.form['insights'] == 'topic1':
        tweets_per_topic = twitter.topics['Count'][0]
        selected_topic = twitter.topics.index[0]
        topic_name = 'topic 1'
    elif request.form['insights'] == 'topic2':
        tweets_per_topic = twitter.topics['Count'][1]
        selected_topic = twitter.topics.index[1]
        topic_name = 'topic 2'
    elif request.form['insights'] == 'topic3':
        tweets_per_topic = twitter.topics['Count'][2]
        selected_topic = twitter.topics.index[2]
        topic_name = 'topic 3'
    elif request.form['insights'] == 'topic4':
        tweets_per_topic = twitter.topics['Count'][3]
        selected_topic = twitter.topics.index[3]
        topic_name = 'topic 4'

    else:
        tweets_per_topic = twitter.topics['Count'][4]
        selected_topic = twitter.topics.index[4]
        topic_name = 'topic 5'

    # Plot WordCloud

    LOW = twitter.create_LOW(selected_topic)
    fig1 = twitter.create_wordcloud(LOW)

    # Plot polarity

    fig2 = twitter.polarity_plot(selected_topic)

    # Plot objectivity

    fig3 = twitter.objectivity_plot(selected_topic)

    return render_template('moreinsights.html', num_of_tweets=app.vars['num_of_tweets'],
                           num_of_tweets_per_topic=tweets_per_topic, topic=topic_name, wordcloud=fig1,
                           polarity=fig2, objectivity=fig3)


@app.route('/error', methods=["GET", "POST"])
def error():
    if request.method == "POST":
        clean_search()
        return redirect('/')
    else:
        return render_template('error.html')


def clean_search():
    app.vars = {}
    return None



if __name__ == '__main__':
    app.run(debug = True)