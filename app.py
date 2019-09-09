from flask import Flask,render_template, request, redirect
import pandas as pd
from twitter_analyzer.Twitter import Twitter
import os
import base64
from rq import Queue
from worker import conn
from redis import Redis

q= Queue('low',connection = conn)


app = Flask(__name__)
app.vars={}
twitter=Twitter('')

@app.route('/',methods=["GET","POST"])
def enter_query():
	if request.method=="GET":
		return render_template('enterquery.html')
	else:
		clean_search()
		return render_template('enterquery.html')

@app.route('/show_analysis',methods=["GET","POST"])
def return_query():
	if request.method=="GET":
		return render_template('returnquery.html', query=app.vars['query'], num_of_tweets=app.vars['num_of_tweets'],
				table=twitter.top_words['words'].to_html(index=False,header=False))
	if request.method=="POST":
    		if request.form['query']=='':
    			return redirect('/')
    		else:
	        	app.vars['query']=request.form['query']
	        
	        	# Get tweets
	        	twitter.query=app.vars['query']
	        
		        # Send task to background worker
		        job=q.enqueue('twitter.get_tweets()')
	 		
			if job.result==None:
				status='Not finished'
			elif job.result!=None:
				status='Finished'

		        #queued_jobs=q.jobs

	        	#data=queued_jobs[0]

		        # Refomat tweets
		        tweets=twitter.reformat_tweets(status)

	        	# Check for empty data frame
		        if tweets.empty == True:
		        	return redirect('/error')
	        

		        # Clean tweets
		        tweets=twitter.clean_and_tokenize()
	
		        app.vars['num_of_tweets']=len(tweets)

	        	# Perform topic modelling
		        tweets=twitter.manualModelling()
	
		        vectorized_tweets=twitter.vectorize_tweets()

	        	matrix=vectorized_tweets['words_matrix']

		        feature_names=vectorized_tweets['feature_names']

		        LDA_model=twitter.fit_LDA(matrix)

	        	twitter.LDA_top_words(LDA_model,feature_names)

	        	return render_template('returnquery.html', query=app.vars['query'],
					num_of_tweets=app.vars['num_of_tweets'],
					table=twitter.top_words['words'].to_html(index=False, header=False))

	             
	        # Plot clustergram
'''
	        twitter.top_labeled_topics()
	        clusters_data=twitter.cluster_text()
	        twitter.create_clustergram(twitter.topics)


	        
	        return render_template('returnquery.html', query=app.vars['query'], 
				num_of_tweets=app.vars['num_of_tweets'],
				table=twitter.top_words['words'].to_html(index=False, header=False), 
				clustergram=twitter.clustergram)
'''

@app.route('/moreinsights', methods=["POST"])
def more_insights():

	# Check what button was clicked on returnquery.html
	if request.form['insights']=='topic1':
		tweets_per_topic = twitter.topics['Count'][0]
		selected_topic = twitter.topics.index[0]
	elif request.form['insights']=='topic2':
		tweets_per_topic = twitter.topics['Count'][1]
		selected_topic = twitter.topics.index[1]
	elif request.form['insights']=='topic3':
		tweets_per_topic = twitter.topics['Count'][2]
		selected_topic = twitter.topics.index[2]
	elif request.form['insights']=='topic4':
		tweets_per_topic = twitter.topics['Count'][3]
		selected_topic = twitter.topics.index[3]
	else:
		tweets_per_topic = twitter.topics['Count'][4]
		selected_topic = twitter.topics.index[4]


	return render_template('moreinsights.html',num_of_tweets=app.vars['num_of_tweets'],num_of_tweets_per_topic=tweets_per_topic, topic=selected_topic)


	# Plot WordCloud
'''
	LOW = twitter.create_LOW(selected_topic)
	fig1 = twitter.create_wordcloud(LOW)

	# Plot polarity
	fig2 = twitter.polarity_plot(selected_topic)

	# Plot objectivity
	fig3 = twitter.objectivity_plot(selected_topic)

	return render_template('moreinsights.html',num_of_tweets=app.vars['num_of_tweets'], 
		num_of_tweets_per_topic=tweets_per_topic, topic=selected_topic,wordcloud=fig1,
		polarity=fig2, objectivity=fig3)
'''
	
@app.route('/error',methods=["GET","POST"])
def error():
	if request.method=="POST":
		clean_search()
		return redirect('/')
	else:
		return render_template('error.html')

def clean_search():
	app.vars={}
	return None

if __name__ == '__main__':
	app.run(debug=True) 
