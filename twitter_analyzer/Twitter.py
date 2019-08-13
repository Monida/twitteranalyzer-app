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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation



#---------------------------------------------------------------------------------
#Functions to get the tweets
#---------------------------------------------------------------------------------

class Twitter:
    def __init__(self,query=''):
        self.query=query
        self.tweets=pd.DataFrame()
        self.creds=self.get_creds()
        self.my_stopwords=self.get_stop_words()
    
    def get_stop_words(self):
        #Set stopwords
        nltk.download('stopwords')
        my_stopwords = nltk.corpus.stopwords.words('english')
        return my_stopwords
    
    def get_creds(self):
        #Credentials
        import json
        with open("creds/twitter_credentials.json","r") as file:
            creds=json.load(file)
        return creds


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
        self.tweets['retweets']=self.tweets['text'].apply(self.find_rt)
        
        # Remove the self.tweets that have a retweet or a link
        for idx, tweet in self.tweets.iterrows():
            if tweet.retweets!=[] or tweet.links!=[]:
                self.tweets.drop(index=idx, inplace=True)
                
        self.tweets.drop(['links','retweets'], axis=1, inplace=True)
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

#---------------------------------------------------------------------------------
#NLP functions
#---------------------------------------------------------------------------------

    def assign_topic(self,token_list,topic_dict,n_highest=1):
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
    def polarity(self,s):
        polarity_val=round(TextBlob(s).sentiment.polarity,2)
        return polarity_val
            
    def subjectivity(self,s):
        subjectivity_val=round(TextBlob(s).sentiment.subjectivity,2)
        return subjectivity_val
    
    def polarity_label(self, s):
        polarity_val=round(TextBlob(s).sentiment.polarity,2)
        if polarity_val <= -0.5:
            return 'negative'
        elif (polarity_val > -0.5 and polarity_val < 0.5):
            return 'neutral'
        else:
            return 'positive'
            
    def subjectivity_label(self, s):
        subjectivity_val=round(TextBlob(s).sentiment.subjectivity,2)
        if subjectivity_val <= 0.4:
            return 'subjective'
        elif (subjectivity_val > 0.4 and subjectivity_val <= 0.6):
            return 'neutral'
        else:
            return 'objective'
    
    
    def clean_and_tokenize(self):
        # Clean and tokenize the tweets
        self.tweets['clean_text']=self.tweets['text'].apply(self.clean_tweet_mtr)
        
        # Drop all the tweets that after cleaning are left with an empty string
        self.tweets=self.tweets[self.tweets.astype(str)['clean_text']!='[]']
        
        tokenizer=TweetTokenizer()
        
        def tokenize_txt(s):
            return tokenizer.tokenize(s)
        
        self.tweets['token_text']=self.tweets['clean_text'].apply(tokenize_txt)
        
        return self.tweets
    
    
    def manualModelling(self):
        # Manually classify the topic
        topics=pd.read_csv('static/keyword_list.csv',sep=',')
        topics_dict={}
        
        for topic in topics.columns:
            topics_dict[topic]=[word for word in list(topics[topic]) if str(word) !='nan']
        
        # Apply the assign_topic function 
        # Find highest matching topic
        self.tweets['topic_1']=self.tweets['token_text'].apply(self.assign_topic,
                    args=(topics_dict,1))
        
        # Find second highest matching topic
        self.tweets['topic_2']=self.tweets['token_text'].apply(self.assign_topic,
                    args=(topics_dict,2))
        
        # Make column topic_1 and topic1_precent. Same with topic 2
        self.tweets['topic_1_percent']=[row[1] for row in self.tweets['topic_1']]
        self.tweets['topic_1']=[row[0] for row in self.tweets['topic_1']]
        
        
        self.tweets['topic_2_percent']=[row[1] for row in self.tweets['topic_2']]
        self.tweets['topic_2']=[row[0] for row in self.tweets['topic_2']]
        
        self.tweets['polarity']=self.tweets['clean_text'].apply(self.polarity)
        self.tweets['polarity_label']=self.tweets['clean_text'].apply(self.polarity_label)
        
        self.tweets['subjectivity']=self.tweets['clean_text'].apply(self.subjectivity)
        self.tweets['subjectivity_label']=self.tweets['clean_text'].apply(self.subjectivity_label)
        
        return self.tweets

    # Inspired by: https://ourcodingclub.github.io/2018/12/10/topic-modelling-python.html#apply
    def vectorize_tweets(self):
        # the vectorizer object will be used to transform text to vector form
        vectorizer = CountVectorizer(max_df=0.9, min_df=15, token_pattern='\w+|\$[\d\.]+|\S+')
        
        # apply transformation
        words_matrix = vectorizer.fit_transform(self.tweets['clean_text']).toarray()
        
        feature_names = vectorizer.get_feature_names()
        
        return {'words_matrix':words_matrix,'feature_names':feature_names}
    
    
    def fit_LDA(self,words_matrix,number_of_topics=10):    
            
        model = LatentDirichletAllocation(n_components=number_of_topics, random_state=0)
        
        model.fit(words_matrix)
        
        return model
    
#---------------------------------------------------------------------------------
# Interpretation functions
# After performing NLP this function extract meaningful information ready to be
# interpretable.
#---------------------------------------------------------------------------------

    
    def top_topics(self,n=5):
        #Tweets: the tweets dataframe 
        #n: top n topics
        topics=self.tweets.groupby('topic_1').count()
    
        topics=topics.rename(columns = {'text':'count'})
        
        topics=topics['count']
        
        topics=topics.sort_values(ascending=False)
        
        top_topics_list=topics[0:n].index.tolist()
        
        return top_topics_list[0:n]
    
    
    #---------------------------------------------------------------------------------
    # Auxiliary functions
    #---------------------------------------------------------------------------------
    
    
    def create_LOW(self):
    #This function takes all the tweets and create a list of words (LOW)
        LOW=[]
        tokenizer=TweetTokenizer()
        for tweet in self.tweets['clean_text']:
            new_token=tokenizer.tokenize(tweet)
            LOW+=new_token
        return LOW

#---------------------------------------------------------------------------------
# Visualization functions
#---------------------------------------------------------------------------------

    def create_figure(self):
        fig = plt.figure()
        plt.hist(self.tweets['polarity'])
        plt.title('Polarity distribution \n -1 negative - 1 possitive')
        return fig
    
    
    def display_topics(self, model, feature_names, no_top_words):
        topic_dict = {}
        for topic_idx, topic in enumerate(model.components_):
            topic_dict["Topic %d words" % (topic_idx)]= ['{}'.format(feature_names[i])
                            for i in topic.argsort()[:-no_top_words - 1:-1]]
            topic_dict["Topic %d weights" % (topic_idx)]= ['{:.1f}'.format(topic[i])
                            for i in topic.argsort()[:-no_top_words - 1:-1]]
        return pd.DataFrame(topic_dict)
    
    def create_wordcloud(self,tokenized_text):
    #This function takes a tokenized text and returns the WordCloud plot
        #Create frequency distribution from tokenized text
        fdist = FreqDist(tokenized_text)
        
        #Filter frequent words; the words that appear more than 3 times
        frequent_words = dict([(k,v) for k,v in fdist.items() if len(k)>3])
        fdist=nltk.FreqDist(frequent_words)
        
        #Create WordCloud
        wcloud = WordCloud().generate_from_frequencies(fdist)
        fig = plt.figure()
        plt.axis("off")
        plt.imshow(wcloud,interpolation='bilinear')
        plt.show()
        return fig


    
