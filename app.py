from flask import Flask,render_template, request, redirect
import pandas as pd

app = Flask(__name__)
app.vars={}

@app.route('/')
def enter_query():
	return render_template('enterquery.html')

@app.route('/show_analysis',methods=["GET","POST"])
def return_query():
	if request.method=="GET":
		clean_search()
		return redirect('/')
	if request.method=="POST":
		app.vars['query']=request.form['query']

		# Send query to analyze
		tweets=analyze_tweets(app.vars['query'])

		app.vars['num_of_tweets']=len(tweets)

		topics=find_topics(tweets)

		return render_template('returnquery.html', query=app.vars['query'], 
			num_of_tweets=app.vars['num_of_tweets'])

def clean_search():
	app.vars['query']=''
	return None

def analyze_tweets(query):
	import twitter_analyzer
	tweets=twitter_analyzer.get_tweets(query)
	tweets=twitter_analyzer.clean_and_tokenize(tweets)
	tweets=twitter_analyzer.manualModelling(tweets)
	return tweets

def find_topics(tweets):
	import twitter_analyzer
	#Identify top topics
	top_topics=twitter_analyzer.top_topics(tweets)

	positives=tweet_polarity(tweets,top_topics,'positive')
	neutrals=tweet_polarity(tweets,top_topics,'neutral')
	negatives=tweet_polarity(tweets,top_topics,'negative')

	topics=pd.DataFrame({'Topics':top_topics,
		'Positive%':positives,
		'Neutral%':neutrals,
		'Negative%':negatives})
	return topics

def tweet_polarity(tweets,topics_list,polarity):
	polarity_percent=[]

	if len(topics_list)==0:
		polarity_percent =[0]*len(topics_list)
	else:
		for topic in topics_list:
			tweets_by_topic=tweets[tweets['topic_1']==topic]
			tweets_by_polar=tweets_by_topic[tweets_by_topic['polarity_label']==polarity]
			if len(tweets_by_polar)!=0:
				polarity_percent.append(round(len(tweets_by_polar)/len(tweets_by_topic),2))
			else:
				polarity_percent.append(0)
	return polarity_percent

if __name__ == '__main__':
	app.run(debug=True) 