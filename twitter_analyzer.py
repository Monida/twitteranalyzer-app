"""
This code is inspired by 
http://www.hristogueorguiev.com/basic-twitter-data-miner-and-data-analysis-
python-twython-twitter-api-pandas-matplotlib/
and
https://ourcodingclub.github.io/2018/12/10/topic-modelling-python.html#top_mod
"""
#%%
from twython import Twython
import pandas as pd
from textblob import TextBlob
import nltk
from nltk.tokenize import TweetTokenizer
import re
import matplotlib.pyplot as plt
from nltk.corpus import webtext
from nltk.probability import FreqDist
from wordcloud import WordCloud


#---------------------------------------------------------------------------------
#Functions to get the tweets
#---------------------------------------------------------------------------------


#Set stopwords
nltk.download('stopwords')
my_stopwords = nltk.corpus.stopwords.words('english')

#Credentials
import json
with open("creds/twitter_credentials.json","r") as file:
    creds=json.load(file)


#Mine tweets from twitter
def MineData(apiobj, query, pagestocollect = 10):

    results = apiobj.search(q=query, include_entities='true',
                             tweet_mode='extended',count='100',
                             result_type='recent',lang='en')

    data = results['statuses']
    i=0
    ratelimit=1
    
    while results['statuses'] and i<pagestocollect: 
        
        if ratelimit < 1: 
            #Rate limit time out needs to be added here in order to
            #collect data exceeding available rate-limit 
            print(str(ratelimit)+'Rate limit!')
            break
        
        mid= results['statuses'][len(results['statuses']) -1]['id']-1

        print(mid)
        print('Results returned:'+str(len(results['statuses'])))
        
        results = apiobj.search(q=query, max_id=str(mid)
                             ,include_entities='true',
                             tweet_mode='extended',count='100',
                             result_type='recent',lang='en')
        
        data+=results['statuses']
        i+=1
        ratelimit = int(apiobj.get_lastfunction_header('x-rate-limit-remaining'))

    return data

def get_tweets(query):  
    # AUTHENTICATE
    twitter = Twython(creds['CONSUMER_KEY'], creds['CONSUMER_SECRET'],  
                        creds['ACCESS_TOKEN'], creds['ACCESS_TOKEN_SECRET'])
    
    # Get the tweets
    dataaccum = MineData(twitter, query,20)
    
    #Reformat data as DataFrame format
    text=[]
    date=[]
    location=[]
    user=[]
    lang=[]
    
    for tweet in dataaccum:
        text.append(tweet['full_text'])
        date.append(tweet['created_at'])
        location.append(tweet['geo'])
        user.append(tweet['user']['id'])
        lang.append(tweet['lang'])
    
    tweets=pd.DataFrame({'text':text,'date':date,'location':location,'lang':lang})
    
    # Remove all entries whose language is different than english
    tweets=tweets[tweets.lang == 'en']
    tweets.drop('lang', axis=1, inplace=True)
    
    # Revome all duplicates
    tweets.drop_duplicates(subset='text',keep='first', inplace=True)
    
    # Make new columns for new extracted information
    tweets['links']=tweets['text'].apply(find_urls)
    tweets['hashtags']=tweets['text'].apply(find_ht)
    tweets['retweets']=tweets['text'].apply(find_rt)
    
    # Remove the tweets that have a retweet or a link
    for idx, tweet in tweets.iterrows():
        if tweet.retweets!=[] or tweet.links!=[]:
            tweets.drop(index=idx, inplace=True)
            
    tweets.drop(['links','retweets'], axis=1, inplace=True)
    return tweets


#---------------------------------------------------------------------------------
# Cleaning functions
#---------------------------------------------------------------------------------
def find_user(s):
    return re.findall(r'@[w\w]+', s)

def find_urls(s):
    urls = re.findall(r'(https?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b)',s)
    return [url[0] for url in urls]

def find_ht(s):
    return re.findall(r'#[w\w]+', s)
                  
def find_spec_char(s):
    return re.findall(r'[^w]',s )

def find_rt(s):
    return re.findall(r'RT ?(@[w\w]+)',s)

def remove_user(s,replace=' '):
    return re.sub(r'@[w\w]+',replace, s)

def remove_urls(s, replace=' '):
    return re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b',replace,s)

def remove_ht(s, portion='all',replace= ' '):
    
    if portion=='all': #if want to remove the whole hashtag
        pattern=r'#[w\w]+'
    elif portion=='hash': #if want to remove  # only
        pattern=r'#'
            
    return re.sub(pattern,replace, s)
                  
def remove_spec_char(s, replace=' '):
    return re.sub(r'[^\w]', replace,s )

def remove_punct(s, replace= ' '):
    return re.sub('[!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@’…”'']+', replace, s)

def remove_double_space(s, replace= ' '):
    return re.sub('\s+', replace, s)

def remove_numb(s, replace= ''):
    return re.sub('([0-9]+)', replace, s)

def remove_emojis(s, replace = ''):
    pass

def spell_checker(s):
    corr_s = TextBlob(s)
    return corr_s.correct()
    
# Cleaning master function
def clean_tweet_mtr(tweet, bigrams=False):
# tweet: a string that contains the text of the tweet
    tweet = remove_user(tweet)
    tweet = remove_urls(tweet)
    tweet = tweet.lower() # lower case
    tweet = remove_punct(tweet)
    tweet = remove_numb(tweet)
    tweet_token_list = [word for word in tweet.split(' ')
                            if word not in my_stopwords] # remove stopwords

    if bigrams:
        tweet_token_list = tweet_token_list+[tweet_token_list[i]+'_'+tweet_token_list[i+1]
                                            for i in range(len(tweet_token_list)-1)]
    tweet = ' '.join(tweet_token_list)
    
    tweet = remove_ht(tweet, portion ='hash')
    tweet = remove_spec_char(tweet)
    tweet = remove_double_space(tweet)
    #tweet = spell_checker(tweet): not ready to be used. It takes to loon to process.
    
    return tweet

#---------------------------------------------------------------------------------
#NLP functions
#---------------------------------------------------------------------------------

def assign_topic(token_list,topic_dict,n_highest=1):
# This function takes a list of tokens and asign them with a topic from a given 
# dictionary. Returns a list of assigned topic and matching percentage 
    percentage_match_dict={}
    
    for topic in topic_dict.keys():
        percentage_match_dict[topic]=0
        
        if len(token_list)!=0:
            for word in token_list:
            
                if word in topic_dict[topic]:
                    percentage_match_dict[topic]+=1
            
    
            percentage_match_dict[topic]=round(percentage_match_dict[topic]/len(token_list),2)
        else:
            percentage_match_dict[topic]=0
            
    percentage_match_series=pd.Series(percentage_match_dict)
    
    matching_topics=percentage_match_series.nlargest(n=n_highest,keep='all')
    
    # From all matching topics, choose the nth_highest assigned topic
    assigned_topic=matching_topics.index[n_highest-1]
    match_percentage=matching_topics[n_highest-1]
    
    return [assigned_topic,match_percentage]

# Sentiment Analysis
def polarity(s):
    polarity_val=round(TextBlob(s).sentiment.polarity,2)
    return polarity_val
        
def subjectivity(s):
    subjectivity_val=round(TextBlob(s).sentiment.subjectivity,2)
    return subjectivity_val

def polarity_label(s):
    polarity_val=round(TextBlob(s).sentiment.polarity,2)
    if polarity_val <= -0.5:
        return 'negative'
    elif (polarity_val > -0.5 and polarity_val < 0.5):
        return 'neutral'
    else:
        return 'positive'
        
def subjectivity_label(s):
    subjectivity_val=round(TextBlob(s).sentiment.subjectivity,2)
    if subjectivity_val <= 0.4:
        return 'subjective'
    elif (subjectivity_val > 0.4 and subjectivity_val <= 0.6):
        return 'neutral'
    else:
        return 'objective'


def clean_and_tokenize(tweets):
    # Clean and tokenize the tweets
    tweets['clean_text']=tweets['text'].apply(clean_tweet_mtr)
    
    # Drop all the tweets that after cleaning are left with an empty string
    tweets=tweets[tweets.astype(str)['clean_text']!='[]']
    
    tokenizer=TweetTokenizer()
    
    def tokenize_txt(s):
        return tokenizer.tokenize(s)
    
    tweets['token_text']=tweets['clean_text'].apply(tokenize_txt)
    
    return tweets


def manualModelling(tweets):
    # Manually classify the topic
    topics=pd.read_csv('static/keyword_list.csv',sep=',')
    topics_dict={}
    
    for topic in topics.columns:
        topics_dict[topic]=[word for word in list(topics[topic]) if str(word) !='nan']
    
    # Apply the assign_topic function 
    # Find highest matching topic
    tweets['topic_1']=tweets['token_text'].apply(assign_topic,
                args=(topics_dict,1))
    
    # Find second highest matching topic
    tweets['topic_2']=tweets['token_text'].apply(assign_topic,
                args=(topics_dict,2))
    
    # Make column topic_1 and topic1_precent. Same with topic 2
    tweets['topic_1_percent']=[row[1] for row in tweets['topic_1']]
    tweets['topic_1']=[row[0] for row in tweets['topic_1']]
    
    
    tweets['topic_2_percent']=[row[1] for row in tweets['topic_2']]
    tweets['topic_2']=[row[0] for row in tweets['topic_2']]
    
    tweets['polarity']=tweets['clean_text'].apply(polarity)
    tweets['polarity_label']=tweets['clean_text'].apply(polarity_label)
    
    tweets['subjectivity']=tweets['clean_text'].apply(subjectivity)
    tweets['subjectivity_label']=tweets['clean_text'].apply(subjectivity_label)
    
    return tweets

# Inspired by: https://ourcodingclub.github.io/2018/12/10/topic-modelling-python.html#apply
def vectorize_tweets(tweets):
    from sklearn.feature_extraction.text import CountVectorizer
    # the vectorizer object will be used to transform text to vector form
    vectorizer = CountVectorizer(max_df=0.9, min_df=15, token_pattern='\w+|\$[\d\.]+|\S+')
    
    # apply transformation
    words_matrix = vectorizer.fit_transform(tweets['clean_text']).toarray()
    
    feature_names = vectorizer.get_feature_names()
    
    return {'words_matrix':words_matrix,'feature_names':feature_names}


def fit_LDA(words_matrix,number_of_topics=10):    

    from sklearn.decomposition import LatentDirichletAllocation
    
    model = LatentDirichletAllocation(n_components=number_of_topics, random_state=0)
    
    model.fit(words_matrix)
    
    return model

#---------------------------------------------------------------------------------
# Interpretation functions
# After performing NLP this function extract meaningful information ready to be
# interpretable.
#---------------------------------------------------------------------------------


def top_topics(tweets,n=5):
    #Tweets: the tweets dataframe 
    #n: top n topics
    topics=tweets.groupby('topic_1').count()

    topics=topics.rename(columns = {'text':'count'})
    
    topics=topics['count']
    
    topics=topics.sort_values(ascending=False)
    
    top_topics_list=topics[0:n].index.tolist()
    
    return top_topics_list[0:n]


#---------------------------------------------------------------------------------
# Auxiliary functions
#---------------------------------------------------------------------------------


def create_LOW(tweets):
#This function takes all the tweets and create a list of words (LOW)
    LOW=[]
    tokenizer=TweetTokenizer()
    for tweet in tweets['clean_text']:
        new_token=tokenizer.tokenize(tweet)
        LOW+=new_token
    return LOW


#---------------------------------------------------------------------------------
# Visualization functions
#---------------------------------------------------------------------------------

def create_figure(tweets):
    fig = plt.figure()
    plt.hist(tweets['polarity'])
    plt.title('Polarity distribution \n -1 negative - 1 possitive')
    return fig


def display_topics(model, feature_names, no_top_words):
    topic_dict = {}
    for topic_idx, topic in enumerate(model.components_):
        topic_dict["Topic %d words" % (topic_idx)]= ['{}'.format(feature_names[i])
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
        topic_dict["Topic %d weights" % (topic_idx)]= ['{:.1f}'.format(topic[i])
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
    return pd.DataFrame(topic_dict)

def create_wordcloud(tokenized_text):
#This function takes a tokenized text and returns the WordCloud plot
    #Create frequency distribution from tokenized text
    fdist = FreqDist(tokenized_text)
    
    #Filter frequent words; the words that appear more than 3 times
    frequent_words = dict([(k,v) for k,v in fdist.items() if len(k)>3])
    fdist=nltk.FreqDist(frequent_words)
    
    #Create WordCloud
    wcloud = WordCloud().generate_from_frequencies(fdist)
    fig = plt.figure()
    plt.imshow(wcloud,interpolation='bilinear')
    return fig
























