# **twitteranalyzer-app**<a id='contents'></a>

<img src="static/twitter_nlp.png">

This repository contains all the files of the [TwitterAnalyzer Web-app](https://twitteranalyzer-app.herokuapp.com/).

Below you can find the description of the content of this repository as well as some links to understand the app's features. 

This readme file has three sections.

 1. [Description](#description)
 2. [App structure](#appstructure)
 3. [Future work](#futurework)


**Note:** if you want to clone, make sure to create the folder 'creds' with the file 'twitter-credentials.json'. [Here](#creds) you can learn how to do it.


## **1. Description**<a id='description'></a>
### **1.1. The app**

The TwitterAnalyzer App is a Web App where the user can enter a search word to learn what people are tweeting about that query. The app uses Natural Language Processing (NLP) techniques to analyze the text of the tweets and come up with meaningful insights. 

The app was built on the [Flask](https://flask.palletsprojects.com/en/1.1.x/) web framework and deployed to [Heroku](https://www.heroku.com/). 

### **1.2. Tweets live streaming**

The App uses [Twython](https://twython.readthedocs.io/en/latest/) to live stream a sample of tweets written in English that people have posted during the 7 days previous to the search, which is the allowed search time frame of the [Standard Twitter API](https://developer.twitter.com/en/pricing.html). 

### **1.3. Twitter data reformatting**
The [search()](https://twython.readthedocs.io/en/latest/usage/basic_usage.html) method of the Twython object returns a list of dictionaries, each representing a tweet. I then transformed this list into a data frame and removed "useless" tweets. 

The "useless" tweets are the ones with links to websites or images, tweets written in languages other than English and duplicates. I do this with a combination of [Pandas](https://pandas.pydata.org/) methods and [regular expressions](https://regexone.com/). 

The reason for removing tweets with links is that usually, one cannot understand those tweets by reading the text alone. Some sort of interpretation of the relationship between the tweet's text and the link is necessary. Therefore, to be able to interpret the meaning of those tweets, a mixed model of NLP and Computer Vision has to be used. Something I hope I could try later on, but for now, I just sticked to the only-text tweets. 

### **1.4. Clean and tokenize**
For the remaining tweets, I use regular expressions to remove all user names, URLs, punctuation, numbers, special characters including emojis (emoji analysis is another feature I will add later on), hashtags symbols and double spaces. I also turn all words into lower cases.

Then with the [TweetTokenizer()](https://www.nltk.org/api/nltk.tokenize.html) class from the [NLTK library](https://www.nltk.org/), I tokenized all words of the clean tweets' texts to create the [bag of words](https://en.wikipedia.org/wiki/Bag-of-words_model) model of each tweet.

### **1.5. Topic Modeling**

Topic modeling is an application of NLP where some texts, in this case tweets, are analyzed to group the information into different topics.

I used three different topic modeling techniques: dictionary-based, LDA and clustering. 

#### **1.5.1. Dictionary-based topic modeling**
Usually, the result of topic modeling is a list of top N words that represent the different topics. Then, human interpretation is needed to analyze those top N words and come up with a label for each topic. Dictionary-based topic modeling allows you to have predefined topic labels to make the top N words more easily interpretable. 

Dictionary-based topic modeling is the simplest way of labeled topic modeling but it requires a lot of manual work to find the topics from a sample of docs and then label them. Each found topic becomes a key of the dictionary and the related words become the values. For this app, once I have the dictionary manually created, the topic of each tweet is assigned according to what related words in the dictionary best match the bag of words representing each tweet. 

The dictionary could also be scrapped from the internet to make the labeling more automatic and sophisticated (yet, another thing I can do to improve the app). These papers [1,2] suggest a interesting way to do it. 

#### **1.5.2. LDA topic modeling**

The Latent Dirichlet Allocation Algorithm (LDA), assigns topics automatically. Each topic is represented by the top N words of that topic. 

To apply LDA we need to follow 3 steps:

1) Words to Matrix: Using the [CountVectorizer()](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) class of the [Scikit-learn Library](https://scikit-learn.org/stable/index.html), we transform all the tweets into a matrix where each row is a tweet, each column is a word in the tweets corpora after removing stop words, and each **ijth** entry is the number of times the jth word appears in the ith tweet.
2) The matrix created in the previous step is the input of the [LDA model](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html) also from Scikit-learn.
3) From the LDA model, we can extract the top N words that represent each topic. 
   
To learn more about how the LDA algorithm works, you can go to this very instructive [video](https://www.youtube.com/watch?v=NYkbqzTlW3w), or read this [paper](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf), which is the paper where David M. Blei proposed it for the first time. 

#### **1.5.3. Clustering**

Clustering is an unsupervised (without need for labels) technique whose objective is to group words, depending on their similarity. Grouping words according to their similarity is another way to finding the top N words of each relevant topic whithin the teets corpus.  

To implement clustering 3 steps are necessary:

1) TF-IDF: term frequency-inverse document frequency is a way to represent the frequency of words in each tweet, but penalizing for those words that are too common to bring value to the document (like the's and of's). The result of TF-IDF transformation is a matrix where the rows are the tweets and the columns are words (the remaining words after removing stop words) and the **ijth** entry represents the TF-IDF value of jth word for the ith tweet. Follow this [link](https://www.youtube.com/watch?v=4vT4fzjkGCQ) for a more thorough explanation of TF-IDF. I used [TfidfVectorizer()](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) from Scikit-learn to implement this section.
2) K-means clustering: the TF-IDF matrix becomes the input to the K-means clustering algorithm, which I implemented using [KMeans()](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html), also from Scikit-learn. This algorithm consists of grouping words around some given K centroids, by finding the closest centroid to each word through an iterative process. This [video](https://www.youtube.com/watch?v=4b5d3muPQmA) contains a graphical description of K-means clustering.
3) Represent clusters in 2 dimensions: The clustering algorithm uses words transformed into vectors in high-dimensional spaces. Using the [t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) (t-distributed Stochastic Neighbor Embedding) and the [cosine_similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html) from Scikit-learn, inspired by [3], we can find the "similarity-distance" between the data points (tweets in vector format) to be able to represent them in 2 dimensions. You can read the [t-SNE user guide](https://scikit-learn.org/stable/modules/manifold.html#t-sne) to better understand how this is done. 

### **1.6. Sentiment Analysis and Polarity**
After identifying the different topics, the user can select any of them and go deeper into each topic to learn whether the tweets in that topic are positive or negative, or whether they are objective or subjective. This is called Sentiment (or Polarity) and Subjectivity Analysis respectively. 

Sentiment and Subjectivity analysis is done using the [TextBlob library](https://textblob.readthedocs.io/en/dev/#). TextBlob finds the polarity and subjectivity value for all the words of each tweet from the polarity and subjectivity [lexicon](https://github.com/sloria/TextBlob/blob/eb08c120d364e908646731d60b4e4c6c1712ff63/textblob/en/en-sentiment.xml). Then it averages the values of all the words to give an overall sentence polarity or subjectivity value. This [link](https://planspace.org/20150607-textblob_sentiment/) describes very clearly how this is done.

### **1.7. Word cloud**
Another thing you can look at in the app after selecting a topic is the word cloud, which is a graphical representation of the frequency of the most common words of that topic. Basically, the most frequent words are larger and the least frequent words are smaller. This is done using the [wordcloud](https://github.com/amueller/word_cloud) library.

### **1.8. Background workers**
When there are some lengthy processes in a Heroku app (like downloading tweets or running analysis on them), it is necessary to perform these processes asynchronously from the main running time, to make the use of the app more efficient. Furthermore, if these lengthy tasks take more than 30 seconds the Heroku server will return a timeout error making the app to crash. 

Therefore, if a task takes more than 30 seconds to run, something called "background worker" needs to be implemented. A background worker is a function that handles all these lengthy tasks. To do so, 
we need to use the [RQ library](http://python-rq.org/) and the [Redis server](https://redislabs.com/lp/python-redis/) for Python. The RQ library is a Python library that allows you to enqueue the lengthly tasks. Once a task is enqueued, the worker will handle it by sending it to the Redis server to be run asynchronously in the background.  If you want to learn how this is implemented, follow this [link](https://devcenter.heroku.com/articles/python-rq). 


[Back to contents](#contents)

## **2. App structure**<a id='appstructure'></a>
This section focuses on explaining all the files needed to make the TwitterAnalyzer App work.
### **2.1 Project folders tree structure**

```bash
└── twitteranalyzer-app
    ├── static
    │   ├── keyword_list.csv
    │   ├── style.css
    │   ├── twitter_nlp.png
    │   └── wordcloud_shape.svg
    ├── templates 
    │   ├── analyzing.html
    │   ├── enterquery.html
    │   ├── error.html
    │   ├── moreinsights.html
    │   └── retrunquery.html
    ├── twitter_analyzer
    │   ├── __init__.py
    │   ├── README.md
    │   ├── Twitter.py
    │   └── wordcloud_shape.png
    ├── creds
    │   └── twitter_credentials.json
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

### **2.2 Project folders description**

* **twitteranalyzer-app:** This is the project folder. All files needed to run the app must be stored here. 

 * **static/:** This static folder contains all files needed by either the HTML files or the app itself to work.
  
    * **style.css:** This file contains the CSS code that gives format to all the HTML templates.
      
    * **keyword_list:** One of the functions of the TwitterAnalyzer App is to find the topics that are being discussed and label them. So far it does that by comparing the most frequent words to the list of words (values) corresponding for each keyword (key) of the dictionary.

    * **twitter_nlp.png:** This is the home page image.
  
  * **templates/:** This folder holds all the HTML templates which give the structure to each page. 

    * **analyzing.html:** This file appears while the program downloads the tweets and analyzes them to indicate to the user that the data is being analyzed and the results will show up shortly.
  
     * **enterquery.html:** This file displays the instructions to use the twitteranalyzer-app. It also contains a form for the user to enter a query as well as a search button.

     * **error.html:** This HTML file is shown when the app presents an error and provides the user with a corresponding message and/or instructions. 

     * **moreinsights:** This HTML is rendered when in the returnquery.html the user clicks on a specific topic to know more insights about it. 

     * **returnquery.html:** This is the main page that is shown after the program analyzes the tweets. It displays a summary of the tweets that were found.
  
  * **twitter_analyzer/:** This folder holds the files of the twitter_analyzer package.

     * **Twitter.py:** This file contains the Twitter class which is used by the app to download, clean, analyze and display the tweets and their corresponding analysis. 
  
     * **__init__:** This file indicates that the directory twitter_analyzer is a python package.
  
  * **creds/:** This folder does not appear in the repo because it is hidden. However if you want the app to work you need to create the folder creds that contains a single file named twitter-credentials.json. <a id='creds'></a>

     * **obtain your twitter credentials:** To create your own credentials, go to the following [link](https://developer.twitter.com/en/docs/basics/authentication/guides/access-tokens) and follow the steps under "Generating access tokens".

     * **twitter-credentials.json:** Once you have your credentials, create the twitter-credential.json file with the following code:

    {"CONSUMER_KEY": "your_consumer_key", "CONSUMER_SECRET": "your_consumer_secret", "ACCESS_TOKEN": "your_access_token", "ACCESS_TOKEN_SECRET": "your_access_token_secret"}

    **Notice that the values of this dictionary are long strings, so remember to put them inside the double quotes.**
  
  * **.gitignore:** This hidden text file contains the files and directories that are not tracked using git.
  
  * **Procfile:** This is a file *with no extension* that contains the commands that need to be executed during the app start-up. Go to the following [link](https://devcenter.heroku.com/articles/procfile) to learn more about it.
  
 * **Procfile.Windows:** This is also a file with no extension and just like Procfile it tells the app what code to run during start-up, but it is an extra file needed when the app is built on Windows. 
  
  * **README.md:** This file (the current file) contains the entire description of the twitteranalyzer-app project.
  
  * **app.py:** This is the main script responsible for running the Flask app. This file is called by Procfile during the start-up of the app.
   
  * **nltk.txt:** This text file is necessary to deploy an app that uses NLTK to Heroku. It contains a reference to the corpora that needs to be downloaded for the app to work. During the app build-up, the corpora are downloaded and installed so that the app doesn't have to download it every time someone uses the app. To learn more about how this is done, go to this [link](https://devcenter.heroku.com/articles/python-nltk).
         
  * **requirements.txt:** Heroku needs this requirements file to be able to install all the libraries that are necessary to be able to run the app. In this file, you will find the name of the python libraries needed for the app to work such as Pandas, Matplotlib, NLTK, Scikit-learn, etc.
  
  * **run_analysis.py:** The Twitteranalyzer App performs some tasks that take a long time to run, such as downloading tweets and their analysis. All these lengthy tasks are listed inside this script so that they can be enqueued and sent to the *background worker*. Read section 1.8 to learn more about how this is done. 
  
  * **runtime.txt:** This file tells Heroku what version of python is needed.
    
  * **worker.py:**   The worker file is the *background worker* function that "listens" to the lengthy tasks that are enqueued and handles them by sending them to the Redis server to be executed asynchronously. 

  [Back to contents](#contents)

  ## **3. Future work**<a id='futurework'></a>

  * Add spell checker to clean and tokenize
  * Improve clustering
  * Improve LDA
  * Improve dictionary based topic modeling
  * Improve data visualization
  * Improve enqueue of jobs to be able to use the app simultaneously

  [Back to contents](#contents)

References:

[1] J.H. Lau, K. Grieser, D. Newman, and T. Baldwin. "Automatic labelling of topic models". In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies. Volume 1,  2011, pages 1536–1545. Association for Computational Linguistics.

[2] S. Bhatia, J. H. Lau, and T. Baldwin, “Automatic labelling of topics with neural embeddings,” in 26th COLING International Conference on Computational Linguistics. 2016, pages 953–963.

[3] A. Kulkarni, A. Shivananda. "Natural Language Processing Recipes". 2019, New York. Apress.