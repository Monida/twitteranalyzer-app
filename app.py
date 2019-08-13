from flask import Flask,render_template, request, redirect
import pandas as pd
from twitter_analyzer.Twitter import Twitter

app = Flask(__name__)
app.vars={}

@app.route('/')
def enter_query():
	clean_search()
	return render_template('enterquery.html')

@app.route('/show_analysis',methods=["GET","POST"])
def return_query():
    if request.method=="GET":
        clean_search()
        return redirect('/')
    
    if request.method=="POST":
        app.vars['query']=request.form['query']
        
        # Analyze tweets
        twitter=Twitter(app.vars['query'])
        tweets=twitter.get_tweets()
        tweets=twitter.clean_and_tokenize()
        tweets=twitter.manualModelling()
        
        app.vars['num_of_tweets']=len(tweets)
        
        topics=find_topics(twitter)
        
        # Plot polarity
        fig=twitter.create_figure(tweets)
        fig.savefig('static/polarity_distribution.png')
        
        # Plot WordCloud
        LOW=twitter.create_LOW(tweets)
        fig=twitter.create_wordcloud(LOW)
        fig.savefig('static/wordcloud.png')
        
        return render_template('returnquery.html', query=app.vars['query'], 
			num_of_tweets=app.vars['num_of_tweets'],table=topics.to_html())

def clean_search():
	app.vars['query']=''
	return None

def find_topics(twitter):
	#Identify top topics
	top_topics=twitter.top_topics()

	positives=tweet_polarity(twitter,top_topics,'positive')
	neutrals=tweet_polarity(twitter,top_topics,'neutral')
	negatives=tweet_polarity(twitter,top_topics,'negative')

	topics=pd.DataFrame({'Topics':top_topics,
		'Positive%':positives,
		'Neutral%':neutrals,
		'Negative%':negatives})
	return topics

def tweet_polarity(twitter,topics_list,polarity):
	polarity_percent=[]

	if len(topics_list)==0:
		polarity_percent =[0]*len(topics_list)
	else:
		for topic in topics_list:
			tweets_by_topic=twitter.tweets[twitter.tweets['topic_1']==topic]
			tweets_by_polar=tweets_by_topic[tweets_by_topic['polarity_label']==polarity]
			if len(tweets_by_polar)!=0:
				polarity_percent.append(round(len(tweets_by_polar)/len(tweets_by_topic),2))
			else:
				polarity_percent.append(0)
	return polarity_percent

if __name__ == '__main__':
	app.run(debug=True) 