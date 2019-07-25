from flask import Flask,render_template, request, redirect

app = Flask(__name__)
app.vars={}
app.vars['query']=''

@app.route('/',methods=["GET","POST"])
def hello_world():
	if request.method=="GET":
		return render_template('enterquery.html')
	if request.method=="POST":
		if app.vars['query']!='':
			clean_search()
		return render_template('enterquery.html')

@app.route('/show_analysis',methods=["GET","POST"])
def return_query():
	if request.method=="GET":
		clean_search()
		return redirect('/')
	if request.method=="POST":
		app.vars['query']=request.form['query']
		return render_template('returnquery.html', query=app.vars['query'])

def clean_search():
	app.vars['query']=''
	return None

if __name__ == '__main__':
	app.run(debug=True) 