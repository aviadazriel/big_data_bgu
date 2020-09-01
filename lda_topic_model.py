from utils import *
from gensim.models.callbacks import PerplexityMetric

#prepair the data 
sfraim_path = 'all_tweets.sframe'
sf = tc.load_sframe(sfraim_path)

#preprocessing
sf = preprocessing(sf)
sf.materialize()
sf =  sf[['text']]

# Clean text:
min_text_len = 2
min_word_len = 3
sf['clean_text'] = sf['text'].apply(lambda x: preprocess_tweet(tweet=x, min_text_len=min_text_len, min_word_len=min_word_len, remove_hashtag=False))                                                                           
sf = sf.dropna()
sf.materialize()

#tokens
sf['tokens'] = sf['clean_text'].apply(lambda x: x.split())
sf = sf[['tokens']]
sf.materialize()

# Create bigrams:
min_count=5
threshold=100
bigram = gensim.models.Phrases(sf['tokens'], min_count=min_count, threshold=threshold)
bigram_mod = gensim.models.phrases.Phraser(bigram)
unigram_and_bigram = [bigram_mod[tweet] for tweet in sf['tokens']]
# Create Dictionary:
id2word = corpora.Dictionary(unigram_and_bigram)

# Term Document Frequency:
corpus = [id2word.doc2bow(tweet) for tweet in unigram_and_bigram]

#Build LDA model
perplexity_logger = PerplexityMetric(corpus=corpus, logger='shell')
coherence_logger  = gensim.models.callbacks.CoherenceMetric(corpus=corpus, coherence="u_mass", logger='shell')

# Config:
num_topics = 4
random_state = 11
update_every = 1
chunksize = 5000
passes=10
alpha='auto'
per_word_topics=True

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics, 
                                           random_state=random_state,
                                           update_every=update_every,
                                           chunksize=chunksize,
                                           passes=passes,
                                           alpha=alpha, callbacks=[coherence_logger],
                                           per_word_topics=per_word_topics)

model_path = './drive/My Drive/big_data/models/lda/lda.model'
lda_model.save(model_path)

#Evaluate
coherence_model_lda = CoherenceModel(model=lda_model, texts=unigram_and_bigram, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('Coherence Score: ', coherence_lda)

# Show topics:
for index, topic in lda_model.show_topics(formatted=False, num_words= 10):
  print('Topic: {} \nWords: {}'.format(index+1, [w[0] for w in topic]))
