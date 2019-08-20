from flask import Flask,render_template, request, redirect
import pandas as pd
from twitter_analyzer.Twitter import Twitter
#from rq import Queue
#from worker import conn


#q= Queue(connection = conn)

#twitter_helper= Twitter()
#results = q.enqueue(twitter_helper.get_punkt())


app = Flask(__name__)
app.vars={}
twitter=Twitter('')

@app.route('/home',methods=["GET","POST"])
def enter_query():
	if request.method=="GET":
		return render_template('enterquery.html')
	else:
		clean_search()
		return render_template('enterquery.html')

@app.route('/show_analysis',methods=["GET","POST"])
def return_query():
    if request.method=="GET":
        return render_template('returnquery.html', query=app.vars['query'], 
				num_of_tweets=app.vars['num_of_tweets'],table=twitter.topics.to_html())
    
    if request.method=="POST":
    	if request.form['query']=='':
    		return redirect('/home')
    	else:
	        app.vars['query']=request.form['query']
	        
	        # Analyze tweets
	        twitter.query=app.vars['query']
	        tweets=twitter.get_tweets()

	        #Check for empty data frame
	        if tweets.empty ==True:
	        	return redirect('/error')
	        tweets=twitter.clean_and_tokenize()
	        tweets=twitter.manualModelling()
	        
	        app.vars['num_of_tweets']=len(tweets)
	        
	        #topics=twitter.top_topics(tweets)
	        twitter.top_topics(tweets)

	        # Plot clustergram
	        clusters_data=twitter.cluster_text()
	        #fig=twitter.create_clustergram(twitter.topics)
	        #fig.savefig('static/clustergram.png')

	        
	        return render_template('returnquery.html', query=app.vars['query'], 
				num_of_tweets=app.vars['num_of_tweets'],table=twitter.topics.to_html())


@app.route('/moreinsights',methods=["POST"])
def more_insights():

    # Plot polarity
    fig=twitter.polarity_plot()
    fig.savefig('static/polarity_distribution.png')
    
    # Plot WordCloud
    LOW=twitter.create_LOW()
    fig=twitter.create_wordcloud(LOW)
    fig.savefig('static/wordcloud.png')

    return render_template('moreinsights.html',num_of_tweets=app.vars['num_of_tweets'])

@app.route('/error',methods=["GET","POST"])
def error():
	if request.method=="POST":
		clean_search()
		return redirect('/home')
	else:
		return render_template('error.html')

def clean_search():
	app.vars={}
	return None

def find_polarities():
	positives=tweet_polarity(twitter,top_topics,'positive')
	neutrals=tweet_polarity(twitter,top_topics,'neutral')
	negatives=tweet_polarity(twitter,top_topics,'negative')
	return None 


def tweet_polarity(twitter,topics_list,polarity):


	topics=pd.DataFrame({'Topics':top_topics,
		'Positive%':positives,
		'Neutral%':neutrals,
		'Negative%':negatives})
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