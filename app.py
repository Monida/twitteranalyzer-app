from flask import Flask,render_template, request

app = Flask(__name__)
app.vars={}

@app.route('/')
def hello_world():
	return render_template('enterquery.html')

@app.route('/show_analysis',methods=["POST"])
def return_query():
	app.vars['query']=request.form['query']
	return app.vars['query']

if __name__ == '__main__':
	app.run(debug=True)