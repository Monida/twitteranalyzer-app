from flask import Flask,render_template

app = Flask(__name__)
app.vars={}

 
@app.route('/')
def hello_world():
	return render_template('enterquery.html')

@app.route('/show_analysis', methods=["GET","POST"])
def return_query():
	if request.method=="POST":
		app.vars['query']=request.form['query']
		return render_template('returnquery.html')
 
if __name__ == '__main__':
	app.run(debug=True)