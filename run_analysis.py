from twitter_analyzer.Twitter import Twitter

def run_analysis(twitter_obj):
	
	# Clean tweets
    
    tweets = twitter_obj.clean_and_tokenize()
    twitter_obj.num_of_tweets = len(tweets)

    # Summary of hashtags

    twitter_obj.hashtag_summary()
    twitter_obj.plot_hts()


    # Perform manual topic modeling

    tweets = twitter_obj.manualModelling()

    # Perform LDA topic modeling

    matrix, terms = twitter_obj.tfidf()
    lda_model = twitter_obj.fit_LDA(matrix)
    twitter_obj.LDA_top_words(lda_model, terms)

    # Create clusters

    twitter_obj.top_labeled_topics()
    twitter_obj.cluster_text()
    twitter_obj.create_clustergram()
    twitter_obj.label_cluster_topics()
    
    return twitter_obj