from flask import Flask,render_template

app = Flask(__name__)
 
@app.route('/')
def hello_world():
	return render_template('enterquery.html')

@app.route('/show_analysis',methods=["POST"])
def return_query():
	return "Sorry this app doesn't analyze tweets yet. But wouldn't it be cool if it did?"

if __name__ == '__main__':
	app.run(debug=True)