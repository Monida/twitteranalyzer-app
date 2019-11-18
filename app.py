from flask import Flask, render_template, request, redirectimport pandas as pdfrom twitter_analyzer.Twitter import Twitterimport osimport base64from rq import Queuefrom worker import connfrom redis import Redisimport timefrom run_analysis import run_analysisq = Queue('low', connection=conn)app = Flask(__name__)app.vars = {}app.vars['analyzing']=Falsetwitter = Twitter('')@app.route('/', methods=["GET", "POST"])def enter_query():    if request.method == "GET":        # Coming from browser        return render_template('enterquery.html')    else:        # Coming from New_Search in return query        q.empty()        twitter = Twitter('')        return render_template('enterquery.html')@app.route('/downloadtweets', methods=["GET","POST"])def download_tweets():    global twitter    if request.method == "POST":        # Coming from enterquery.html        if request.form['query'] == '':            return redirect('/')                    else:            if app.vars['analyzing']==False:                twitter.query = request.form['query']                # Get_tweets                data = twitter.get_tweets()                # Reformat tweets                tweets = twitter.reformat_tweets(data)                if tweets.empty:                    return redirect('/error')                job = q.enqueue(run_analysis, twitter, job_id = 'analysis')            app.vars['analyzing'] = True            return render_template('analyzing.html', query = twitter.query)@app.route('/show_analysis', methods=["GET", "POST"])def return_query():    global twitter    if request.method == "GET":        # Coming from moreinsights.html        job=q.fetch_job('analysis')        twitter=job.result        return render_template('returnquery.html', query=twitter.query,                                   num_of_tweets=twitter.num_of_tweets,                                   hts_table = twitter.top_hts.to_html(),                                   topics_table = twitter.topics.to_html(),                                   topwords_table = twitter.top_words['words'].to_html(index=False, header=False),                                   clustergram=twitter.clustergram)    if request.method == "POST":        # Coming from analyzing.html        job = q.fetch_job('analysis')        if job.result == None:            return render_template('analyzing.html', query = twitter.query)        else:            twitter=job.result            app.vars['analyzing']=False            return render_template('returnquery.html', query=twitter.query,                                   num_of_tweets=twitter.num_of_tweets,                                   hts_table = twitter.top_hts.to_html(),                                   topics_table = twitter.topics.to_html(),                                   topwords_table = twitter.top_words['words'].to_html(index=False, header=False),                                   clustergram=twitter.clustergram)@app.route('/moreinsights', methods=["POST"])def more_insights():    global twitter# Check what button was clicked on returnquery.html    job = q.fetch_job('analysis')    twitter = job.result    if request.form['insights'] == 'topic 1':        tweets_per_topic = twitter.topics['Count'][0]        selected_topic = twitter.topics.index[0]        topic_name = 'topic 1'    elif request.form['insights'] == 'topic 2':        tweets_per_topic = twitter.topics['Count'][1]        selected_topic = twitter.topics.index[1]        topic_name = 'topic 2'    elif request.form['insights'] == 'topic 3':        tweets_per_topic = twitter.topics['Count'][2]        selected_topic = twitter.topics.index[2]        topic_name = 'topic 3'    elif request.form['insights'] == 'topic 4':        tweets_per_topic = twitter.topics['Count'][3]        selected_topic = twitter.topics.index[3]        topic_name = 'topic 4'    else:        tweets_per_topic = twitter.topics['Count'][4]        selected_topic = twitter.topics.index[4]        topic_name = 'topic 5'    # Plot WordCloud    LOW = twitter.create_LOW(selected_topic)    fig1 = twitter.create_wordcloud(LOW)    # Plot polarity    fig2 = twitter.polarity_plot(selected_topic)    # Plot objectivity    fig3 = twitter.objectivity_plot(selected_topic)    return render_template('moreinsights.html', num_of_tweets=twitter.num_of_tweets,                           num_of_tweets_per_topic=tweets_per_topic, topic=topic_name, wordcloud=fig1,                           polarity=fig2, objectivity=fig3)@app.route('/error', methods=["GET", "POST"])def error():    if request.method == "POST":        clean_search()        return redirect('/')    else:        return render_template('error.html')def clean_search():    app.vars = {}    app.vars['analyzing']=False    return Noneif __name__ == '__main__':    app.run(debug = True)