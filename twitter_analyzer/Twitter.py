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

class Twitter:
    def __init__(self,query=''):
        self.query=query
        self.tweets=pd.DataFrame()
    
    def get_stop_words(self):
        #Set stopwords
        nltk.download('stopwords')
        my_stopwords = nltk.corpus.stopwords.words('english')
        return my_stopwords
    
    def get_creds(self):
        #Credentials
        import json
        with open("creds/twitter_credentials.json","r") as file:
            self.creds=json.load(file)

'''
    #Mine tweets from twitter
    def MineData(self,apiobj, pagestocollect = 10):
    
        results = apiobj.search(q=self.query, include_entities='true',
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
            
            results = apiobj.search(q=self.query, max_id=str(mid)
                                 ,include_entities='true',
                                 tweet_mode='extended',count='100',
                                 result_type='recent',lang='en')
            
            data+=results['statuses']
            i+=1
            ratelimit = int(apiobj.get_lastfunction_header('x-rate-limit-remaining'))
    
        return data
    
    def get_tweets(self):  
        # AUTHENTICATE
        twitter_obj = Twython(self.creds['CONSUMER_KEY'], self.creds['CONSUMER_SECRET'],  
                            self.creds['ACCESS_TOKEN'], self.creds['ACCESS_TOKEN_SECRET'])
        
        # Get the tweets
        dataaccum = self.MineData(twitter_obj,20)
        
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
        
        self.tweets=pd.DataFrame({'text':text,'date':date,'location':location,'lang':lang})
        
        # Remove all entries whose language is different than english
        self.tweets=self.tweets[self.tweets.lang == 'en']
        self.tweets.drop('lang', axis=1, inplace=True)
        
        # Revome all duplicates
        self.tweets.drop_duplicates(subset='text',keep='first', inplace=True)
        
        # Make new columns for new extracted information
        self.tweets['links']=self.tweets['text'].apply(self.find_urls)
        self.tweets['hashtags']=self.tweets['text'].apply(self.find_ht)
        self.tweets['reself.tweets']=self.tweets['text'].apply(self.find_rt)
        
        # Remove the self.tweets that have a retweet or a link
        for idx, tweet in self.tweets.iterrows():
            if tweet.reself.tweets!=[] or tweet.links!=[]:
                self.tweets.drop(index=idx, inplace=True)
                
        self.tweets.drop(['links','reself.tweets'], axis=1, inplace=True)
        return self.tweets
    
    #---------------------------------------------------------------------------------
    # Cleaning functions
    #---------------------------------------------------------------------------------
    def find_user(self,s):
        return re.findall(r'@[w\w]+', s)
    
    def find_urls(self,s):
        urls = re.findall(r'(https?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b)',s)
        return [url[0] for url in urls]
    
    def find_ht(self,s):
        return re.findall(r'#[w\w]+', s)
                      
    def find_spec_char(self,s):
        return re.findall(r'[^w]',s )
    
    def find_rt(self,s):
        return re.findall(r'RT ?(@[w\w]+)',s)
    
    def remove_user(self,s,replace=' '):
        return re.sub(r'@[w\w]+',replace, s)
    
    def remove_urls(self,s, replace=' '):
        return re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b',replace,s)
    
    def remove_ht(self,s, portion='all',replace= ' '):
        
        if portion=='all': #if want to remove the whole hashtag
            pattern=r'#[w\w]+'
        elif portion=='hash': #if want to remove  # only
            pattern=r'#'
                
        return re.sub(pattern,replace, s)
                      
    def remove_spec_char(self,s, replace=' '):
        return re.sub(r'[^\w]', replace,s )
    
    def remove_punct(self,s, replace= ' '):
        return re.sub('[!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@’…”'']+', replace, s)
    
    def remove_double_space(self,s, replace= ' '):
        return re.sub('\s+', replace, s)
    
    def remove_numb(self,s, replace= ''):
        return re.sub('([0-9]+)', replace, s)
    
    def remove_emojis(self,s, replace = ''):
        pass
    
    def spell_checker(self,s):
        corr_s = TextBlob(s)
        return corr_s.correct()
        
    # Cleaning master function
    def clean_tweet_mtr(self,tweet, bigrams=False):
    # tweet: a string that contains the text of the tweet
        tweet = self.remove_user(tweet)
        tweet = self.remove_urls(tweet)
        tweet = tweet.lower() # lower case
        tweet = self.remove_punct(tweet)
        tweet = self.remove_numb(tweet)
        tweet_token_list = [word for word in tweet.split(' ')
                                if word not in self.my_stopwords] # remove stopwords
    
        if bigrams:
            tweet_token_list = tweet_token_list+[tweet_token_list[i]+'_'+tweet_token_list[i+1]
                                                for i in range(len(tweet_token_list)-1)]
        tweet = ' '.join(tweet_token_list)
        
        tweet = self.remove_ht(tweet, portion ='hash')
        tweet = self.remove_spec_char(tweet)
        tweet = self.remove_double_space(tweet)
        #tweet = spell_checker(tweet): not ready to be used. It takes to loon to process.
        
        return tweet
'''