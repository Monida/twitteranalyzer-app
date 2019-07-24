from flask import Flask,render_template

app = Flask(__name__)
 
@app.route('/')
def hello_world():
	return render_template('enterquery.html')

@app.route('/show_analysis')
def return_query():
	return "Hello again!"
 
if __name__ == '__main__':
	app.run(debug=True)