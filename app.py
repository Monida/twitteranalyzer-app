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
		app.vars['num_of_tweets']=analyze_tweets(app.vars['query'])

		return render_template('returnquery.html', query=app.vars['query'], num_of_tweets=app.vars['num_of_tweets'])

def clean_search():
	app.vars['query']=''
	return None

def analyze_tweets(query):
	import twitter_analyzer
	tweets=twitter_analyzer.get_tweets(query)
	num_of_tweets=len(tweets)
	return num_of_tweets


if __name__ == '__main__':
	app.run(debug=True) 