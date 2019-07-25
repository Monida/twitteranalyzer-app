from flask import Flask,render_template, request

app = Flask(__name__)
app.vars={}

@app.route('/',methods=["GET","POST"])
def hello_world():
	if request.method=="GET":
		return render_template('enterquery.html')
	if request.method=="POST":
		if app.vars['query']!='':
			app.vars['query']==''
		return render_template('enterquery.html')

@app.route('/show_analysis',methods=["POST"])
def return_query():
	app.vars['query']=request.form['query']
	return render_template('returnquery.html', query=app.vars['query'])

def clean_search():
	app.vars['query']=''
	return None

if __name__ == '__main__':
	app.run(debug=True) 