from twitter_analyzer.Twitter import Twitter

def run_analysis(twitter_obj):
	
	# Clean tweets
    
    tweets = twitter_obj.clean_and_tokenize()
    twitter_obj.num_of_tweets = len(tweets)

    # Summary of hashtags

    hashtags = twitter_obj.hashtag_summary()

    # Perform manual topic modeling

    tweets = twitter_obj.manualModelling()

    # Perform LDA topic modeling

    vectorized_tweets = twitter_obj.vectorize_tweets()
    matrix = vectorized_tweets['words_matrix']
    feature_names = vectorized_tweets['feature_names']
    lda_model = twitter_obj.fit_LDA(matrix)
    twitter_obj.LDA_top_words(lda_model, feature_names)

    # Create clustergram

    twitter_obj.top_labeled_topics()
    clusters_data = twitter_obj.cluster_text()
    twitter_obj.create_clustergram()
    
    return twitter_obj