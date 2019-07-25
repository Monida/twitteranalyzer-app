from flask import Flask,render_template

app = Flask(__name__)
app.vars={}

@app.route('/')
def hello_world():
	return render_template('enterquery.html')

@app.route('/show_analysis',methods=["POST"])
def return_query():
	return render_template('returnquery.html')

if __name__ == '__main__':
	app.run(debug=True)