# **twitteranalyzer-app**

<img src="static/twitter_nlp.png">


**-------------------------------------------------FILE UNDER CONSTRUCTION------------------------------------------------------**


This repository contains all the files of the [TwitterAnalyzer](link:https://twitteranalyzer-app.herokuapp.com/) Web-app.

Below you can find the description of the content of this repository. 


## **1. Description**
### **1.1. The app**

The TwitterAnalyzer App is a Wep App where the user can enter a search word (query) to learn what people are tweeting about in relation to that query. 

### **1.2. Tweets live streaming**

The App uses [Twython](link:https://twython.readthedocs.io/en/latest/) to live stream a sample of tweets written in English, that people have posted during the past 7 days (according to the Twitter API policy for the [Standard API](link:https://developer.twitter.com/en/pricing.html). 

### **1.3. Twitter data reformatting**
The [search()](link:https://twython.readthedocs.io/en/latest/usage/basic_usage.html) method of the Twython objet returns a list of dictionaries, each representing a tweet. I transformed this list into a data frame and removing "useless" tweets. 

The "useless" tweets are the ones with links to websites or images, tweets written in languages other than English, and duplicates. I do this with a combination of Pandas methods and regular expressions. 

The reason for removing tweets with links is that usually one can understand the tweet by only reading the text. Some sort of interpretation of the relationship between the tweet's text and the link need to be done. Therfore, To be able to interpret the meaning of those tweets, a mixed model of NLP and Computer Vision is needed. Something I hope I could try later on, but for now, I just stick with the only-text tweets. 

### **1.4. Clean and tokenize**
For the remaining tweets I use regular expressions to remove all user names, urls, punctuation, numbers, special characters including emojis (emoji analysis is another feature I will add later on), hashtags (just the symbol) and double spaces. I also turn all words into lower cases.

Then with the TweetTokenizer() class from the NLTK library I tokenize all words of the clean tweets' texts.

### **1.5. Topic Modeling**

Topic modeling is an aplication of NLP where some texts, in this case tweets, are analyzed to group the information into different topics.

I used three different topic modeling techniques: dictionary-based, LDA and clustering. 

#### **1.5.1. Dictionary-based**
Dictionary-based topic modeling is the simplest way of topic modeling but it requires a lot of manual work to find the topics from a sample of docs and then lable them. Each found topic becomes a key of the dictionary and the related words become the values. The dictionary could also be scrapped from the internet to make it more sofisticated (yet, another thing I have to improve).

Once the dictionary is ready, the topic to each tweet is assigned according to what related words match best the bag of words representing each tweet. 

#### **1.5.2. LDA**

The Latent Dirichlet Allocation Algorightm (LDA), assigns topics authomatically. Each topic is represented by the top N words of that topic. 

To apply LDA we need to follow 3 steps:

1) Words to Matrix: Using the [CountVectorizer()](link:https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) method of the Sci-Kit Learn Library, we transformed all the tweets into a matrix where each row is a tweet, each column is a word in the tweets corpora, and each matrix entry **ij** is a one or a zero depending on the presence of the word of the jth column being on the tweet of the ith row.
2) The matrix created in the previous step is the input of the [LDA model](link:https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html) also from Sci-Kit learn.
3) From the LDA model we can extract the top N words that represent each topic. 

#### **1.5.3. Clustering**


### **1.6. Sentiment Analysis and Polarity**


### **1.7. Word cloud**


### **1.8. Background workers**
When there are some lengthy processes in a Heroku app (like downloading tweets or running analysis on them), it is necesary to perform these processes asynchronously from the main running time to make the use of the app more efficient. Furthermore, if these lengthy tasks take more than 30 seconds the Heroku server will return a timeout error making the app to crash. 

Therefore, if a taks takes more than 30 seconds to run, something called "background worker" needs to be implemented. A background worker is a function that handles all this lengthy tasks. We need to use the [RQ library](link:http://python-rq.org/) and the [Redis server](https://redislabs.com/lp/python-redis/) for Python. The RQ library is a Python library that allows you to enqueue the lengthly tasks. Once a task is enqueued, the worker will handle it by sending it to the Redis server to be run asynchronously in the background.  If you want to learn how this is implemented follow this link: https://devcenter.heroku.com/articles/python-rq. 

## **2. App structure**
### **2.1 Project folders tree structure**

```bash
└── twitteranalyzer-app
    ├── static
    │   ├── keyword_list.csv
    │   ├── style.css
    │   ├── twitter_nlp.png
    │   ├── wordcloud_shape.svg
    ├── templates 
    │   ├── analyzing.html
    │   ├── enterquery.html
    │   ├── error.html
    │   ├── moreinsights.html
    │   ├── retrunquery.html
    ├── twitter_analyzer
    │   ├── __init__.py
    │   ├── README.md
    │   ├── Twitter.py
    │   ├── wordcloud_shape.png
    ├── venv
    │   ├── bin
    │   ├── share
    │   ├── pip-selfchekc.json
    │   ├── pyvenv.cfg
    ├── .gitignore
    ├── app.py
    ├── nltk.txt
    ├── Procfile
    ├── Procfile.windows
    ├── README.md
    ├── requirements.txt
    ├── run_analysis.py
    ├── runtime.txt
    └── worker.py
```

## **2.2 Project folders description**

* **twitteranalyzer-app:** This is the project folder. All files needed to run the app must be stored here. 

 * **static/:** This static folder contains all files needed by either the html files or the app itself to work.
  
    * **style.css:** This file contains the CSS code that gives format to all the html templates.
      
    * **keyword_list:** One of the functions of the TwitterAnalyzer is to find the topics that are being discussed and label them. So far it does that by comparing the most frecuent words to the list of words (values) corresponding for each keyword (key) of the dictionary.

    * **twitter_nlp.png:** This is the home page image.
  
  * **templates/:** This folder holds all the html templates, whic gives the structure to each page. 

    * **analyzing.html:** This file appears while the program downloads the tweets and analyze them to indicate the user that the data is being analyzed and the results will show up shortly.
  
     * **enterquery.html:** This file displays the instructions to use the twitteranalyzer-app. It also contains the form for the user to enter a query as well as a search button.

     * **error.html:** This html file is showed when the app presents an error and provides the user with a corresponding message and/or instructions. (Not implemented yet)

     * **moreinsights:** This html is rendered when in the returnquery.html the user clicks on a specific topic to know more insights about it. 

     * **returnquery.html:** This is the main page that is shown after the program analyzes the tweets. It displays a summary about the tweets that were found.
  
  * **twitter_analyzer/:** This folder holds the files of the twitter_analyzer package.

     * **Twitter.py:** This file contains the Twitter class which is used by the app to download, clean, analyze and display the tweets and their corresponding analysis. 
  
     * **__init__:** This file indicates that the directory twitter_analyzer is a python package.
  
  * **venv/:** this folder contains all the information of the virtual environment where the Flask app was created. 
  
  * **.gitignore:** This hidden text file contains the files and directories that are not tracked using git.
  
  * **Procfile:** This is a file *with no extension* that contains the commands that need to be execuded during the app startup. Go to the following link to learn more about it: https://devcenter.heroku.com/articles/procfile.
  
 * **Procfile.Windows:** This is also a file with no extension and just like Procfile it tells the app what code to run during start up, but it is an extra file needed when the app is built on Windows. 
  
  * **README.md:** This file (the current file) contains the entire description of the twitteranalyzer-app project.
  
  * **app.py:** This is the main script responsible for running the Flask app. This file is called by Procfile during the start up of the app.
   
  * **nltk.txt:** This text file is necesary to deploy to Heroku an app that uses NLTK. It contains a refenrence to the corpora that needs to be downloaded for the app to work. During the app build up of the app, the corpora is downloaded and installed so that the app doesn't have to download it everytime someone uses the app. To learn more about how this is done go to the link: https://devcenter.heroku.com/articles/python-nltk.
         
  * **requirements.txt:** Heroku needs this requirements file to be able to install all the libraries that are necesary to be able to run the app. In this file you will find the name of the python libraries needed for the app to work such as pandas, matplotlib, nltk, etc.
  
  * **run_analysis.py:** The Twitteranalyzer App performs some tasks that take a long time to run, such as downloading tweets and their analysis. All these lengthy tasks are listed inside this script so that they can be enqueued and sent to the *background worker*. Read section 1.8 to learn more about how this is done. 
  
  * **runtime.txt:** This file tells Heroku what version of python is needed.
    
  * **worker.py:**   The worker file is the *background worker* function that "listens" to the lengthy tasks that are enqueued and handles them by sending them to the Redis server to be excecuted asynchronously. 

