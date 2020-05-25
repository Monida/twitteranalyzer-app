"""
This code is inspired by 
http://www.hristogueorguiev.com/basic-twitter-data-miner-and-data-analysis-
python-twython-twitter-api-pandas-matplotlib/
and
https://ourcodingclub.github.io/2018/12/10/topic-modelling-python.html#top_mod
"""
#%%
import pandas as pd
import numpy as np
from twython import Twython
from textblob import TextBlob
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import webtext
from nltk.probability import FreqDist
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import re
from wordcloud import WordCloud
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import base64
import io
from PIL import Image
import os.path
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

#---------------------------------------------------------------------------------
# Functions to get the tweets
#---------------------------------------------------------------------------------

class Twitter:
    def __init__(self,query=''):
        self.query=query
        self.tweets=pd.DataFrame()
        self.creds=self.get_creds()
        self.my_stopwords=self.get_stop_words()
        self.my_stopwords.append('amp')
        self.my_stopwords += ["'d", "'ll", "'re", "'s", "'ve", 'could', 'might', 'must', "n't", 'need', 'sha', 'wo', 'would', "ca", "na"]
        self.topics=pd.DataFrame()
        self.num_of_tweets=0
    
    def get_stop_words(self):
        # Set stopwords
        my_stopwords = nltk.corpus.stopwords.words('english')
        return my_stopwords
    
    def get_creds(self):
        # Credentials
        import json
        with open("creds/twitter_credentials.json","r") as file:
            creds=json.load(file)
        return creds

    
    def get_tweets(self, query='', pagestocollect = 20):
        # This method downloads the tweets from twitter and returns a list of dictionaries
        # each representing a tweet.
        # This function was inspired from: http://www.hristogueorguiev.com/
        # basic-twitter-data-miner-and-data-analysis-python-twython-twitter-api-pandas-matplotlib/
        # The get_tweets function is based on the Standard search API form Twitter

        # AUTHENTICATE
        twitter_obj = Twython(self.creds['CONSUMER_KEY'], self.creds['CONSUMER_SECRET'],  
                            self.creds['ACCESS_TOKEN'], self.creds['ACCESS_TOKEN_SECRET'])
        if query=='':
            query=self.query
    
        results = twitter_obj.search(q=query, include_entities='true',
                                 tweet_mode='extended',count='100',
                                 result_type='recent',lang='en')
    
    # The following code follows this logic
    # https://developer.twitter.com/en/docs/tweets/timelines/guides/working-with-timelines
        data = results['statuses']
        i=0
        ratelimit=1
        
        while results['statuses'] and i<pagestocollect: 
            
            if ratelimit < 1: 
                # Rate limit time out needs to be added here in order to
                # collect data exceeding available rate-limit 
                print(str(ratelimit)+'Rate limit!')
                break
            
            mid= results['statuses'][len(results['statuses']) -1]['id']-1
    
            results = twitter_obj.search(q=query, max_id=str(mid)
                                 ,include_entities='true',
                                 tweet_mode='extended',count='100',
                                 result_type='recent',lang='en')
            
            data+=results['statuses']
            i+=1
            ratelimit = int(twitter_obj.get_lastfunction_header('x-rate-limit-remaining'))
    
        return data
    

    def reformat_tweets(self,data):
        # This method reformat the data as DataFrame format
        # data: a list of dictionaries, each representinga  tweet, returned by the get_tweets() method
        
        text=[]
        date=[]
        location=[]
        user=[]
        lang=[]
        
    
        for tweet in data:
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

    def replace_negations(self, s, replace = ' not'):
        return re.sub(r"n't", replace, s)
    
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
                      
    def remove_query(self,s, replace= ' '):
        return re.sub(self.query,replace,s)

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
        tweet = self.replace_negations(tweet)
        tweet = self.remove_punct(tweet)
        tweet = self.remove_numb(tweet)

        # Remove 'no' and 'not' grom my_stopwords so that they remain in the tokenized
        # tweet. This will help sentiment analysis. 

        self.my_stopwords.remove('no')
        self.my_stopwords.remove('not')

        tweet_token_list = [word for word in tweet.split(' ')
                                if word not in self.my_stopwords] # remove stopwords

        self.my_stopwords.append('no')
        self.my_stopwords.append('not')
    
        if bigrams:
            # Create bigrams
            tweet_token_list = tweet_token_list+[tweet_token_list[i]+'_'+tweet_token_list[i+1]
                                                for i in range(len(tweet_token_list)-1)]
        tweet = ' '.join(tweet_token_list)
        
        tweet = self.remove_ht(tweet, portion ='all')
        tweet = self.remove_spec_char(tweet)
        tweet = self.remove_double_space(tweet)
        tweet = self.lemmatize_tweet(tweet)
        #tweet = spell_checker(tweet): not ready to be used. It takes to long to process.
        
        return tweet

#---------------------------------------------------------------------------------
#Preprocessiong functions
#---------------------------------------------------------------------------------

    # Lemmatize
    # Code inspired by:
    # %%https://medium.com/@gaurav5430/using-nltk-for-lemmatizing-sentences-c1bfff963258

    # function to convert nltk tag to wordnet tag
    def nltk_tag_to_wordnet_tag(self, nltk_tag):
        #The adjectives are not lemmatized so that "worst" is not changed to "bad"
        #hence affecting the polarity estimation
        if nltk_tag.startswith('J'):
            pass
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:          
            return None

    def lemmatize_tweet(self, tweet):
        lemmatizer = WordNetLemmatizer()
        #tokenize the tweet and find the POS tag for each token
        nltk_tagged = nltk.pos_tag(nltk.word_tokenize(tweet))  
        #tuple of (token, wordnet_tag)
        wordnet_tagged = map(lambda x: (x[0], self.nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
        lemmatized_tweet = []
        for word, tag in wordnet_tagged:
            if tag is None:
                #if there is no available tag, append the token as is
                lemmatized_tweet.append(word)
            else:        
                #else use the tag to lemmatize the token
                lemmatized_tweet.append(lemmatizer.lemmatize(word, tag))
        return " ".join(lemmatized_tweet)


    def tokenize(self,text):
        #This function takes a cleaned text, then tokenizes to
        #retrieve a list of tokens
        tokens = [word for sent in nltk.sent_tokenize(text) for
        word in nltk.word_tokenize(sent)]
        return tokens

    def create_tweets_list(self):
        self.tweets_list=self.tweets['clean_text'].tolist()
        return self.tweets_list


#---------------------------------------------------------------------------------
#NLP functions
#---------------------------------------------------------------------------------
        
    def clean_and_tokenize(self):
        # This method cleans and tokenizes the tweets that the reformat_tweets() method returns
        self.tweets['clean_text']=self.tweets['text'].apply(self.clean_tweet_mtr)
        
        # Drop all the tweets that after cleaning are left with an empty string
        self.tweets=self.tweets[self.tweets.astype(str)['clean_text']!='[]']
        
        tokenizer=TweetTokenizer()
        
        def tokenize_txt(s):
            return tokenizer.tokenize(s)
        
        self.tweets['token_text']=self.tweets['clean_text'].apply(tokenize_txt)
        
        return self.tweets


    def tfidf(self):
        # This funciton fermorm TF-IDF vectorization. It takes the clean tweets from
        # tweets['clean_text'] and then returns the tfidf-matrix and tfidf_terms.
        # You need to run clean_and_tokenize before running this function.
        self.create_tweets_list()

        tfidf_vectorizer=TfidfVectorizer(max_df=0.7, max_features=250,
            min_df=15, stop_words=self.my_stopwords,
            use_idf=True, tokenizer = self.tokenize,
            analyzer = 'word',
            ngram_range = (1,4))

        self.tfidf_matrix = tfidf_vectorizer.fit_transform(self.tweets_list)
        self.tfidf_terms = tfidf_vectorizer.get_feature_names()

        return self.tfidf_matrix, self.tfidf_terms


    def manualModelling(self):
        # Manually classify the topic using a dictionary where the keys are the topics and the 
        # values are the related words.
        topics=pd.read_csv('static/keyword_list.csv',sep=',')
        
        # Transform topics dataframe into a dictionary
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
        
        self.tweets['objectivity']=self.tweets['clean_text'].apply(self.objectivity)
        self.tweets['objectivity_label']=self.tweets['clean_text'].apply(self.objectivity_label)
        
        return self.tweets


    def assign_topic(self,token_list,topic_dict,n_highest=1):
    # This function takes a list of tokens and asign them with a topic from a given 
    # dictionary. Returns a list of one assigned topic and the corresponding matching percentage 
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
            
    def objectivity(self,s):
        objectivity_val=round(TextBlob(s).sentiment.subjectivity,2)
        return objectivity_val
    
    def polarity_label(self, s):
        polarity_val=round(TextBlob(s).sentiment.polarity,2)
        if polarity_val <= -0.5:
            return 'Negative'
        elif (polarity_val > -0.5 and polarity_val < 0.5):
            return 'Neutral'
        else:
            return 'Positive'
            
    def objectivity_label(self, s):
        objectivity_val=round(TextBlob(s).sentiment.subjectivity,2)
        if objectivity_val <= 0.4:
            return 'Objective'
        elif (objectivity_val > 0.4 and objectivity_val <= 0.6):
            return 'Neutral'
        else:
            return 'Subjective'

    def label_cluster_topics(self):
        # This function finds the topic labels that are more likely for each topic.
        # Need to run manualModeling() and cluster_text() first.

        sorted_tweets = self.tweets.sort_values(by='cluster', axis=0, inplace=False)
        aggregated_tweets = sorted_tweets.groupby(['cluster','topic_1']).count()

        def percentages(x):
            return '%d' % (x) + '%'

        self.cltr_topics = pd.DataFrame()
        for i in range(5):
            tmp_df = pd.DataFrame()
            count = 'count' +  str(i+1)
            topic = 'topic' + str(i+1)
            percent = 'percent' + str(i+1)
            tmp_df[count] = aggregated_tweets.iloc[aggregated_tweets.index.get_level_values(0) == i]['text']
            tmp_df[topic] = tmp_df.index.get_level_values('topic_1').values
            total_tweets = sum(tmp_df[count].values)
            tmp_df = tmp_df[tmp_df[count]>=10]
            tmp_df[percent] = round(tmp_df[count]*100/total_tweets,2)
            tmp_df.sort_values(by=percent,inplace=True, ascending=False)
            tmp_df[percent] = tmp_df[percent].apply(percentages)
            tmp_df.drop(count, axis=1, inplace=True)
            tmp_df.reset_index(inplace=True, drop=True)

            self.cltr_topics=pd.concat([self.cltr_topics,tmp_df],axis=1,ignore_index=False, sort=False)

        #for i in range(5):
        #    topic = 'topic' + str(i+1)
        #    percent = 'percent' + str(i+1)
        #    self.cltr_topics[topic].fillna(' ', inplace = True)
        #    self.cltr_topics[percent].fillna('0%', inplace = True)

        return self.cltr_topics
   
    
    def fit_LDA(self,words_matrix,number_of_topics=5):
        # This method fits the LDA model to the words_matrix, given, to find number of topics provided.
        # words_matrix: is the resulf of running the method tfidf()
            
        model = LatentDirichletAllocation(n_components=number_of_topics, random_state=0)
        
        model.fit(words_matrix)
        
        return model

    def cluster_text(self):
        # This function takes the tweets and clusters them. It adds two new columns
        # tweets['cluster']: the number of cluster it corresponds to
        # tweets['rank']: rank useful to identify the tweets during clustering
        ranks=[]
        for i in range(1,len(self.tweets_list)+1):
            ranks.append(i)
        
        # Need to run tfidf first to compute self.tfidf_matrix and self.tweets_list
        # Clustering using K-means
        num_clusters = 5

        km = KMeans(n_clusters=num_clusters)
        km.fit(self.tfidf_matrix)

        # final clusters
        self.clusters = km.labels_.tolist()
        tweets_data = { 'rank': ranks, 'complaints': self.tweets_list,'cluster': self.clusters }
        clusters_df = pd.DataFrame(tweets_data, index = [self.clusters],columns = ['rank', 'cluster'])
        self.tweets['cluster'] = clusters_df['cluster'].values
        return self.tweets

    
#---------------------------------------------------------------------------------
# Interpretation functions
# After performing NLP this function extract meaningful information ready to be
# interpretable.
#---------------------------------------------------------------------------------

    
    def top_labeled_topics(self, n=5):
        # This function displays the label of the topics that were assigned 
        # usingManualModelling and asign_Topic
        # n: top n topics
        self.topics=self.tweets.groupby('topic_1').count()
    
        self.topics=self.topics.rename(columns = {'text':'Count'})
        
        self.topics=pd.DataFrame(self.topics['Count'])

        total=sum(self.topics.values)[0]

        self.topics['Percentage']=round(self.topics['Count']/total*100,1)

        self.topics['Percentage']=self.topics['Percentage'].apply(lambda p: str(p)+'%')

        self.topics=self.topics.sort_values(by='Count', ascending=False)
        
        self.topics=self.topics.iloc[0:5]

        del self.topics.index.name

        return self.topics
    
    
    #---------------------------------------------------------------------------------
    # Auxiliary functions
    #---------------------------------------------------------------------------------

    def hashtag_summary(self):
        hts_dict = {}
        for tweet in self.tweets['hashtags']:
            hts = self.find_ht(str(tweet))
            for ht in hts:
                if ht not in hts_dict:
                    hts_dict[ht] = 1
                else:
                    hts_dict[ht] +=1
        hts_df = pd.DataFrame({'hashtags':list(hts_dict.keys()),'count':list(hts_dict.values())})
        hts_df.sort_values(by=['count'], inplace=True, axis=0, ascending= False)
        self.top_hts=hts_df[0:5]
        return self.top_hts


    def create_LOW(self,topic):
    #This function takes all the tweets and create a list of words (LOW)
        LOW=[]
        tokenizer=TweetTokenizer()
        filtered_tweets=self.tweets[self.tweets['topic_1']==topic]
        for tweet in filtered_tweets['clean_text']:
            new_token=tokenizer.tokenize(tweet)
            LOW+=new_token
        return LOW

    #---------------------------------------------------------------------------------
    # Visualization functions
    #---------------------------------------------------------------------------------

    def plot_hts(self):
    # This function plots all the top hashtags found after running hashtags_summary()

        img=io.BytesIO()
        x = np.arange(len(self.top_hts))
        values = self.top_hts['count'].values
        labels = self.top_hts['hashtags'].values
        plt.bar(x,values, color = '#03cea4')
        plt.xticks(x, labels, rotation = 'vertical')
        plt.tight_layout()
        plt.savefig(img,format = 'png')
        img.seek(0)
        figure_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

        self.hashtags_plot = 'data:image/png;base64,{}'.format(figure_url)
        return self.hashtags_plot


    def LDA_top_words(self, model, feature_names, num_top_words=10):
        # This function returns the topics (top words) found using the fit_LDA method
        self.top_words={}
        words=[]
        weights=[]
        
        for topic_idx, topic in enumerate(model.components_):
            words.append(['{}'.format(feature_names[i])
                            for i in topic.argsort()[:-num_top_words - 1:-1]])
            weights.append(['{:.1f}'.format(topic[i])
                            for i in topic.argsort()[:-num_top_words - 1:-1]])
        
        self.top_words['words']=pd.DataFrame(words)
        self.top_words['weights']=pd.DataFrame(weights)

        return self.top_words


    def polarity_plot(self,topic):
    # Figure returning inpired by:
    # https://technovechno.com/creating-graphs-in-python-using-matplotlib-flask-framework-pythonanywhere/

        polarities=self.tweets[self.tweets['topic_1']==topic]

        polarities=self.tweets.groupby('polarity_label').count()

        polarities=polarities.rename(columns = {'text':'count'})

        polarities=pd.DataFrame(polarities['count'])

        polarities['polarity_label']=polarities.index

        img=io.BytesIO()
        
        x = np.arange(3)
        values = polarities['count'].values
        labels = list(polarities['polarity_label'])

        def percentages(x,pos):
            return '%d' % (x/10) + '%'

        formatter = FuncFormatter(percentages)
        
        fig, ax = plt.subplots()
        ax.yaxis.set_major_formatter(formatter)
        plt.bar(x, values,color='#03cea4')
        plt.xticks(x, labels)
        plt.savefig(img,format='png')
        img.seek(0)
        figure_url=base64.b64encode(img.getvalue()).decode()
        plt.close()

        return 'data:image/png;base64,{}'.format(figure_url)


    def objectivity_plot(self,topic):

        objectivities=self.tweets[self.tweets['topic_1']==topic]

        objectivities=self.tweets.groupby('objectivity_label').count()

        objectivities=objectivities.rename(columns = {'text':'count'})

        objectivities=pd.DataFrame(objectivities['count'])

        objectivities['objectivity_label']=objectivities.index

        img = io.BytesIO()

        x = np.arange(3)
        values = objectivities['count'].values
        labels = ['Subjetive','Neutral','Objective']

        def percentages(x,pos):
            return '%d' % (x/10) + '%'

        formatter = FuncFormatter(percentages)

        fig, ax = plt.subplots()
        ax.yaxis.set_major_formatter(formatter)
        plt.bar(x, values, color='#03cea4')
        plt.xticks(x, labels)
        plt.savefig(img,format='png')
        img.seek(0)
        figure_url=base64.b64encode(img.getvalue()).decode()
        plt.close()

        return 'data:image/png;base64,{}'.format(figure_url)

    def create_wordcloud(self,tokenized_text):
    #This function takes a tokenized text and returns the WordCloud plot
        #Create frequency distribution from tokenized text
        fdist = FreqDist(tokenized_text)
        
        #Filter frequent words; the words that appear more than 3 times
        frequent_words = dict([(k,v) for k,v in fdist.items() if len(k)>3])
        fdist=nltk.FreqDist(frequent_words)
        
        #Create WordCloud
        wcloud_mask=np.array(Image.open('static/twitter_nlp.png'))
        wcloud = WordCloud(background_color='white',mask=wcloud_mask).generate_from_frequencies(fdist)
        img=io.BytesIO()
        fig = plt.figure()
        plt.axis("off")
        plt.imshow(wcloud,interpolation='bilinear')
        plt.savefig(img,format='png')
        img.seek(0)
        figure_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        return 'data:image/png;base64,{}'.format(figure_url)


    # Inspired by Natural Language Process Recipes, Akshay Kulkarni & Adarsha Shivananda, 2019.
    def create_clustergram(self):
        similarity_distance = 1 - cosine_similarity(self.tfidf_matrix)
        # Convert two components as we're plotting points in a two-dimensional plane
        tsne =  TSNE(n_components=2, perplexity = 40, random_state=1)
        pos = tsne.fit_transform(similarity_distance)
        #Shape (n_components, n_samples)
        xs, ys = pos[:, 0], pos[:, 1]
        #Set up colors per clusters using a dict

        cluster_colors = {0: '#DBF4AD', 1: '#A9E190', 2: '#CDC776',
                          3: '#A5AA52', 4: '#818D92'}
        
        #Set up cluster names using a dict. Later change it by authomatic topic modelling

        cluster_names = {0: 'Cluster 1',
                         1: 'Cluster 2',
                         2: 'Cluster 3',
                         3: 'Cluster 4',
                         4: 'Cluster 5'}
        
        # Plot clustergram
        # Create data frame that has the result of the MDS and the cluster
        df = pd.DataFrame(dict(x=xs, y=ys, label=self.clusters))
        groups = df.groupby('label')
        # Set up plot
        img = io.BytesIO()
        fig, ax = plt.subplots(figsize=(17, 9)) # set size
        for name, group in groups:
            ax.plot(group.x, group.y, marker='o', linestyle='', ms=20,
                label=cluster_names[name], color=cluster_colors[name],
                mec='none')
            ax.set_aspect('auto')
            ax.tick_params(\
                           axis= 'x',
                           which='both',
                           bottom='off',
                           top='off',
                           labelbottom='off')
            ax.tick_params(\
                           axis= 'y',
                           which='both',
                           left='off',
                           top='off',
                           labelleft='off')
            ax.legend(numpoints=1)
        plt.savefig(img,format='png')
        img.seek(0)
        figure_url = base64.b64encode(img.getvalue()).decode()
        self.clustergram='data:image/png;base64,{}'.format(figure_url)

        return self.clustergram





    
