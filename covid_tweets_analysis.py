from utils import *
from gensim.models.callbacks import PerplexityMetric
from gensim.models import Word2Vec
from gensim.test.utils import get_tmpfile
from gensim.models.callbacks import CallbackAny2Vec

corona_keyword = ["corona", "#corona", "coronavirus", "#coronavirus", "covid", "#covid", "covid19", "#covid19", "covid-19", "#covid-19",
"sarscov2", "#sarscov2", "sars cov2", "sars cov 2", "covid_19", "#covid_19", "#ncov", "ncov", "#ncov2019", "ncov2019",
"2019-ncov", "#2019-ncov", "pandemic", "#pandemic" "#2019ncov", "2019ncov", "quarantine", "#quarantine", "flatten the curve",
"flattening the curve", "#flatteningthecurve", "#flattenthecurve", "hand sanitizer", "#handsanitizer", "#lockdown", "lockdown",
"social distancing", "#socialdistancing", "work from home", "#workfromhome", "working from home", "#workingfromhome", "ppe",
"n95", "#ppe", "#n95", "#covidiots", "covidiots", "herd immunity", "#herdimmunity", "pneumonia", "#pneumonia", "chinese virus",
"#chinesevirus", "wuhan virus", "#wuhanvirus", "kung flu", "#kungflu", "wearamask", "#wearamask", "wear a mask", "vaccine", "vaccines",
"#vaccine", "#vaccines", "corona vaccine", "corona vaccines", "#coronavaccine", "#coronavaccines", "face shield", "#faceshield",
"face shields", "#faceshields", "health worker", "#health worker", "health workers", "#healthworkers", "#stayhomestaysafe",
"#coronaupdate", "#frontlineheroes", "#coronawarriors", "#homeschool", "#homeschooling", "#hometasking", "#masks4all",
"#wfh", "wash ur hands", "wash your hands", "#washurhands", "#washyourhands", "#stayathome", "#stayhome", "#selfisolating",
"self isolating", "bars closed", "restaurants closed"]


def word_vector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in tokens:
        try:
            vec += model_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:  # handling the case where the token is not in vocabulary
            continue
    if count != 0:
        vec /= count
    return vec
    
class EpochLogger(CallbackAny2Vec):
     '''Callback to log information about training'''

     def __init__(self):
         self.epoch = 0
     def on_epoch_begin(self, model):
         print("Epoch #{} start".format(self.epoch))
     def on_epoch_end(self, model):
         self.epoch += 1
         
def get_topics_per_doc(ldamodel, corpus):
  
  topics_df = pd.DataFrame()
  l = [ldamodel.get_document_topics(daily_tokens, minimum_probability=0) for daily_tokens in corpus]
  cols = []
  for row in tqdm(l):
    dist = []
    col = []
    for j, (topic_num, prop_topic) in enumerate(row):
      col.append('Topic '+str(topic_num))
      dist.append(prop_topic)
    
    cols.append(col)
    topics_df = topics_df.append(pd.Series(dist),ignore_index=True)

  assert len(set(list(map(' '.join, cols))))==1
  topics_df.columns = cols[0]
  topics_df['Topic'] = topics_df.idxmax(axis=1)
  return topics_df, cols[0]
  
# Load data:
"""
Depends on the program running: covid_spread.py
"""

covid_spread_state_gb_path = "drive/My Drive/Colab Notebooks/Big Data/final_project/Output/covid_spread_state_gb.csv"
corona_per_dates_gb_path = "drive/My Drive/Colab Notebooks/Big Data/final_project/Output/corona_per_dates_gb.csv"
covid_spread_state_gb = pd.read_csv(covid_spread_state_gb_path)
corona_per_dates_gb = pd.read_csv(corona_per_dates_gb_path)
input_url = './drive/My Drive/big_data/tables/covid Tweets.csv'
covid_df = pd.read_csv(input_url,low_memory=False)
print(f'covid df shape {covid_df.shape[0]}')

"""Data preprocessing"""

# Data preprocessing:
cols = ['created_at','id','source', 'text','in_reply_to_status_id','in_reply_to_user_id','user.id','State_abbv']
covid_df = preprocessing(covid_df, specific_columns = cols)

"""Graph show covid cases and Number of Tweets Per Day"""

# Covid tweets count over time:
tweet_count_gb = covid_df.groupby(by='date').count()['id'].reset_index()
tweet_count_gb = tweet_count_gb.rename(columns={'id':'count'})
tweet_count_gb['date'] = tweet_count_gb['date'].astype(str)
tweet_count_with_covid_gb = tweet_count_gb.merge(corona_per_dates_gb, how='left', on='date')
tweet_count_with_covid_gb['covid cases'] = tweet_count_with_covid_gb['covid cases'].fillna(value=0)

t = tweet_count_with_covid_gb['date']
count = tweet_count_with_covid_gb['count']
cases = tweet_count_with_covid_gb['covid cases']
deaths = tweet_count_with_covid_gb['covid deaths']

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(
    go.Scatter(x=t, y=savgol_filter(cases,21,5), name="Covid Cases", line=dict(color='red', width=2, dash='dot') ),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=t, y=savgol_filter(count,11,3), name="Covid Tweets Count", line=dict(color='green', width=2)),
    secondary_y=True,
)

fig.update_layout(
    title_text="Covid tweets count over time in USA"
)
fig.update_xaxes(title_text="Day")
fig.update_yaxes(title_text="<b>Covid-19 Cases</b>", secondary_y=False)
fig.update_yaxes(title_text="<b>Covid-19 Tweets Count</b>", secondary_y=True)
fig.show()

"""Clean Text and Features Extract"""
#Clean text:
min_text_len = 2
min_word_len = 3
covid_df['clean_text'] = covid_df['text'].apply(lambda x: preprocess_tweet(tweet=x, min_text_len=min_text_len,
                                                                                   min_word_len=min_word_len, remove_hashtag=False))
covid_df = covid_df[~covid_df['clean_text'].isna()]

covid_df['clean_text_w2v'] = covid_df['text'].apply(lambda x: preprocess_tweet(tweet=x, min_text_len=min_text_len,min_word_len=min_word_len))   
covid_df = covid_df[~covid_df['clean_text_w2v'].isna()]

#Extract sentiment:
covid_df['sentiment'] = covid_df['clean_text_w2v'].progress_apply(lambda x: get_sentiment(x))

"""Map show sentiment over time by County (Hover Mouse: Covid-Cases and Covid-Deaths)"""
# USA covid tweets sentiment:
covid_per_country_df = covid_df.groupby(by=['date','State_abbv']).mean().reset_index()
covid_per_country_df['date'] = covid_per_country_df['date'].apply(lambda x: str(x).replace(" 00:00:00",''))
covid_per_country_gb = covid_per_country_df.merge(covid_spread_state_gb, on=['date','State_abbv'], how='left')
covid_per_country_gb['covid cases'] = covid_per_country_gb['covid cases'].fillna(value=0) 
covid_per_country_gb['covid deaths'] = covid_per_country_gb['covid deaths'].fillna(value=0)
covid_per_country_gb['covid cases'] = covid_per_country_gb['covid cases'].astype(int)
covid_per_country_gb['covid deaths'] = covid_per_country_gb['covid deaths'].astype(int)
covid_per_country_gb = covid_per_country_gb[['date','State_abbv','sentiment','covid cases','covid deaths']]
px.choropleth(covid_per_country_gb, locations='State_abbv',
              locationmode='USA-states', scope="usa", hover_data=['covid cases','covid deaths'], color='sentiment',animation_frame="date", color_continuous_scale="ylgn", title="Covid tweets sentiment per state")

"""Topic Model - use Gensim LDA"""

#prepare data

# Create doc per day:
covid_df['tokens_t'] = covid_df['clean_text'].apply(lambda x: x.split())
covid_df['tokens_w'] = covid_df['clean_text_w2v'].apply(lambda x: x.split())

# Create bigrams:
min_count=5
threshold=100
covid_bigram = gensim.models.Phrases(covid_df['tokens_t'], min_count=min_count, threshold=threshold)
covid_bigram_mod = gensim.models.phrases.Phraser(covid_bigram)
covid_unigram_and_bigram = [covid_bigram_mod[tweet] for tweet in covid_df['tokens_t']]
# Create Dictionary:
covid_id2word = corpora.Dictionary(covid_unigram_and_bigram)

# Term Document Frequency:
covid_corpus = [covid_id2word.doc2bow(tweet) for tweet in covid_unigram_and_bigram]

# Build LDA model:
perplexity_logger = PerplexityMetric(corpus=covid_corpus, logger='shell')

# Config:
num_topics = 4
random_state = 11
update_every = 1
chunksize = 5000
passes=10
alpha='auto'
per_word_topics=True

warnings.filterwarnings("ignore",category=DeprecationWarning)
lda_covid_model = gensim.models.ldamodel.LdaModel(corpus=covid_corpus,
                                           id2word=covid_id2word,
                                           num_topics=num_topics, 
                                           random_state=random_state,
                                           update_every=update_every,
                                           chunksize=chunksize,
                                           passes=passes,
                                           alpha=alpha, callbacks=[perplexity_logger],
                                           per_word_topics=per_word_topics)


"""Load Pretrained Models (Optional)"""
lda_model_3_topic_path = './drive/My Drive/big_data/models/lda/lda_covid3/lda_covid3.model'
lda_model_4_topic_path = './drive/My Drive/big_data/models/lda/lda_covid_4/lda_covid_4.model'
lda_model_5_topic_path = './drive/My Drive/big_data/models/lda/lda_covid_5/lda.model'

lda_covid_model_3_topics = gensim.models.ldamodel.LdaModel.load(lda_model_3_topic_path)
lda_covid_model_4_topics = gensim.models.ldamodel.LdaModel.load(lda_model_4_topic_path)
lda_covid_model_5_topics = gensim.models.ldamodel.LdaModel.load(lda_model_5_topic_path)

models ={'LDA 3 Topics':lda_covid_model_3_topics , 'LDA 4 Topics': lda_covid_model_4_topics, 'LDA 5 Topics' : lda_covid_model_5_topics}

#Evaluate 
warnings.filterwarnings("ignore",category=DeprecationWarning)
for modelLDA_key in models:
  print(f'model {modelLDA_key}:')
  coherence_model_lda = CoherenceModel(model=models[modelLDA_key], texts=covid_unigram_and_bigram, dictionary=covid_id2word, coherence='c_v')
  coherence_lda = coherence_model_lda.get_coherence()
  print('Coherence Score: ', coherence_lda)

lda_covid_model = lda_covid_model_4_topics

"""We Choose Model with 4 Topics By Maximum Coherence Score"""

# Show topics:
for index, topic in lda_covid_model.show_topics(formatted=False, num_words= 10):
  print('Topic: {} \nWords: {}'.format(index+1, [w[0] for w in topic]))

"""Analysis Topic Per Tweet"""

covid_df['covid_corpus'] = covid_corpus

# Extract topic per day:
covid_topics_df, covid_topic_cols = get_topics_per_doc(ldamodel=lda_covid_model, corpus=covid_corpus)

covid_data_and_topics_df = concat_dfs_by_col(list_of_dfs=[covid_df_sample, covid_topics_df])

# show Topic Distrubution
D = dict(Counter(covid_data_and_topics_df['Topic']))
rcParams['figure.figsize'] = 10, 10
ax = sns.barplot(x= list(D.keys()) , y=list(D.values()))

"""Word Embedding - Load Pre Train"""

epoch_logger = EpochLogger()
model_path = '/content/drive/My Drive/big_data/models/word2vec/half_all_17M_TRAINABLE/model_w2v.wv'
model_w2v = Word2Vec.load(model_path)

print(model_w2v.wv.most_similar(positive= "trump", topn=15))

# Get similar words:
print(model_w2v.wv.most_similar(positive= "corona", topn=15))

"""tweet2vec"""

df_filter_uncommon_topics = covid_data_and_topics_df[(covid_data_and_topics_df['Topic'] != 'Topic 0') ]
df_filter_uncommon_topics = df_filter_uncommon_topics.reset_index()

tokens = df_filter_uncommon_topics['tokens_w']
# Convert tweet to vector:
wordvec_arrays = np.zeros((len(tokens), 300)) 
for i in range(len(tokens)):
    wordvec_arrays[i,:] = word_vector(tokens[i], 300)
wordvec_df = pd.DataFrame(wordvec_arrays)

"""Visualization tweets using PCA"""

# Visualization tweets using PCA:
pca = PCA(n_components=2)
pca_result = pca.fit_transform(wordvec_arrays)
df_filter_uncommon_topics['pca-one'] = pca_result[:,0]
df_filter_uncommon_topics['pca-two'] = pca_result[:,1] 
fig = px.scatter(df_filter_uncommon_topics, x="pca-one", y="pca-two", color="Topic")
fig.show()
