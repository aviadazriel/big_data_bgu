"""
Since we had a lot of data to analyze we are looking for the optimal method to analyze the data. 
For this task we worked many days to learn the best way to analyze such an amount of data. We tried tools like Spark, Turicreate and Dataframe.
We deside to analyze we used Amazon Web Services, the instances we use are: ml.m5.12xlarge and ml.m5.4xlarge were the large one has 192 GB RAM and 48 CPU units.
This code is relative to s3 format.
We analyze Twitter conversations to identify topics and trends in the Twitter data, we use topic modeling, specifically, LDA model.
"""

from utils import *
import dask.dataframe as dd
import multiprocessing
from dask.distributed import progress

def load_data_from_s3(s3_client, bucket, key):
    
    response = s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix =key,
                MaxKeys=100 )

    # get file names:
    file_list = []
    for f in response['Contents']:
        if "csv" in f['Key']:
            file_list.append("s3://"+str(bucket)+"/"+f['Key'])

    # Load files:
    dfs = []
    for file in tqdm(file_list):
        try:
          df = pd.read_csv(file, low_memory=False)
        except:
          try:
            df = pd.read_csv(file, engine='python')
          except:
            try:
              df = pd.read_csv(file,sep='\t',encoding="ISO-8859-1")
            except:
                try:
                    df = pd.read_csv(file, ep='\\t',lineterminator='\\r', engine='python', header='infer')
                except:
                  print("filed to load file: "+ file)

        dfs.append(df)

    # Merge data:
    df = concat_dfs_by_row(list_of_dfs=dfs)
    print("\nDF shape is {s}".format(s=str(df.shape[0])))

    return df

def save_pickle_file_to_s3(s3_client, df, bucket, key, obj_name):
    
    serializedMyData = cPickle.dumps(df)
    s3_client.put_object(Bucket=bucket,Key=obj_name, Body=serializedMyData)
    
def save_df_to_s3_as_pickle_chuncks(s3_client, bucket, df, file_name, chunk_size, key):
    
    df_list = np.array_split(df, chunk_size)
    
    for d in tqdm(range(len(df_list))):
        name = file_name +"_" + str(d)
        chunck_df = df_list[d]
        save_pickle_file_to_s3(s3_client=s3_client, df=chunck_df, bucket=bucket, key=key, obj_name=str(name)+'.obj')

def compute_coherence_values(dictionary, corpus, texts, limit, random_state, update_every, passes, alpha, per_word_topics, chunksize, start=2, step=3):

    
    import warnings
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    coherence_values = []
    model_list = []
    for num_topics in tqdm(range(start, limit, step)):
        
        model = gensim.models.LdaMulticore(workers=mp.cpu_count()-2,corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=random_state,
                                                chunksize=chunksize, passes=passes, per_word_topics=per_word_topics)

        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values
  
  
  # aws:
import boto3
from s3fs.core import S3FileSystem

# Configs:
s3_resource = boto3.resource('s3')
s3_client = boto3.client('s3')
s3_file = S3FileSystem()
bucket='bigdatabgu'

# Load data from aws:
random_tweets_df = load_data_from_s3(s3_client=s3_client, bucket='bigdatabgu', key='Random Tweets')
covid_spread_state_gb = pd.read_csv("s3://bigdatabgu/covid_spread_state_gb.csv")

# Data preprocessing:
cols = ['created_at','id','text','in_reply_to_status_id','in_reply_to_user_id','user.id','State_abbv']
random_tweets_df = preprocessing(random_tweets_df, specific_columns = cols)

# Tag corona tweets:
random_tweets_df['covid_tweet'] = random_tweets_df['text'].progress_apply(lambda x: is_relevent_tweet(text=x,words_list=corona_keyword))

# Create pie chart:
create_pie_chart(df=random_tweets_df, group_by_col='covid_tweet', count_col='id', title='Covid tweets distribution out of randomly sampled tweets')

# Clean text:
min_text_len = 2
min_word_len = 3
ddf= dd.from_pandas(random_tweets_df, npartitions= mp.cpu_count()-2)
random_tweets_df['clean_text'] = ddf.map_partitions(lambda df: df.apply(lambda x: preprocess_tweet(tweet=x['text'],
                                                               min_text_len=min_text_len,
                                                               min_word_len=min_word_len,
                                                               keywords_list=corona_keyword,
                                                               keywords_replacement_word='covid',
                                                               remove_hashtag=False),axis=1)).compute(scheduler='processes')

# Tweet length:
random_tweets_df['num_words'] = random_tweets_df['clean_text'].apply(lambda x: len(x.split()))
plot_histogram(df=random_tweets_df, column='num_words', title='Tweets length distribution')

random_tweets_df['tokens'] = random_tweets_df['clean_text'].progress_apply(lambda x: x.split())
"""
LDA - Latent Dirichlet Allocation (Topic Model)
For this task we used in gensim model LDA Multicore on 10 procent of the dull data to figure out what is the best number of topic the model should trained on the full data
After it, we deside to train a turicreate model on the full data (the turicreate model wes show much better performance)

For more information:
LDA Gensim Model:
https://radimrehurek.com/gensim/models/ldamodel.html

turicreate LDA topic model:
https://apple.github.io/turicreate/docs/api/generated/turicreate.topic_model.create.html
"""

# Grid search on num of topics

# Sample from df:
sub_random_tweets_df = random_tweets_df.sample(frac=0.1)

# Create bigrams:
min_count=50
threshold=300
random_tokens = sub_random_tweets_df['tokens']
random_tweets = sub_random_tweets_df['clean_text']
random_bigram = gensim.models.Phrases(random_tokens, min_count=min_count, threshold=threshold)
random_bigram_mod = gensim.models.phrases.Phraser(random_bigram)
random_unigram_and_bigram = [random_bigram_mod[tweets_tokens] for tweets_tokens in tqdm(random_tokens)]

# Create Dictionary:
random_id2word = corpora.Dictionary(random_unigram_and_bigram)

# Term Document Frequency:
random_corpus = [random_id2word.doc2bow(tweet) for tweet in random_unigram_and_bigram]

# Train LDA model and use grid search to find the best num of topics:
start=20
limit=150
step=10
chunksize = 50000
random_state=11
update_every=1
passes=10
alpha='auto'
per_word_topics=True

model_list, coherence_values = compute_coherence_values(dictionary=random_id2word,
                                                        corpus=random_corpus,
                                                        texts=random_unigram_and_bigram,
                                                        start=start,
                                                        limit=limit,
                                                        step=step,
                                                        random_state=random_state,
                                                        update_every=update_every,
                                                        passes=passes,
                                                        alpha=alpha,
                                                        per_word_topics=per_word_topics,
                                                        chunksize=chunksize)

# Plot the coherence values:
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()

# Get best model:
model_selected = np.argmax(coherence_values)
best_model = model_list[model_selected]
model_topics = best_model.show_topics(formatted=False)

# Visualize the topics:
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(model_topics, random_corpus, covid_id2word)

# Train LDA on all data
# Convert to sframe:
random_tweets_sf = tc.SFrame(random_tweets_df)
# Create BOW:
docs = tc.text_analytics.count_words(random_tweets_sf['clean_text'])

# Train a topic model:
import multiprocessing as mp
tc.config.set_runtime_config('TURI_DEFAULT_NUM_PYLAMBDA_WORKERS', mp.cpu_count()-2)
num_topics = 130
num_iterations = 100
topic_model = tc.topic_model.create(docs, num_topics=num_topics, num_iterations=num_iterations)

# Save model:
topic_model.save("s3://bigdatabgu/topic_model_dframe_130_topics_100_prc")

# Topics generated from LDA Model:
words_per_topic_df = topic_model.get_topics().to_dataframe()
topics = words_per_topic_df.groupby(by='topic')['word'].apply(np.array).reset_index()
topics = topics.rename(columns={"word": "Words", "topic": "Topic"})
topics['Topic'] = topics['Topic']+1

print("Topics generated from LDA Model:")
topics.head(10)

# Predict topic:
random_tweets_sf['Topic'] = topic_model.predict(docs)
random_tweets_sf['docs'] = docs

sample_random_tweets_sf = random_tweets_sf.sample(fraction=0.05)
# Predict topic probability:
topic_probability_array = topic_model.predict(sample_random_tweets_sf['docs'],output_type='probability')

topic_dict = defaultdict(list)
for probabilities in tqdm(topic_probability_array):
  topic_num = 1
  for topic_prob in probabilities:
    topic_dict['Topic '+str(topic_num)].append(topic_prob)
    topic_num+=1
    
# Distribution of Topics in the Corpora:
import turicreate.aggregate as agg
topic_gb = random_tweets_sf.groupby(key_column_names='Topic',operations={'id': agg.COUNT()})
topic_gb = topic_gb.to_dataframe()
topic_gb = topic_gb.rename(columns={"id": "count"})
ax = sns.barplot(x="Topic",y='count', data=topic_gb).set_title('Distribution of Topics in the Corpora')
plt.show()

# Get daily topic distribution:
topic_prob_gb = topic_prob_sf.groupby(key_column_names='date',operations={cols[0]: agg.AVG(cols[0])})
for col in cols[1:]:
  gb = topic_prob_sf.groupby(key_column_names='date',operations={col: agg.AVG(col)})
  topic_prob_gb = topic_prob_gb.add_columns(gb[[col]])
  
topic_prob_gb = topic_prob_gb.to_dataframe()

topic_prob_gb['date_str'] = topic_prob_gb['date'].astype(str)
daily_topic_dist_df = topic_prob_gb.merge(corona_per_dates_gb, left_on='date_str', right_on='date', how='left')
daily_topic_dist_df['covid cases'] = daily_topic_dist_df['covid cases'].fillna(value=0)
daily_topic_dist_df = daily_topic_dist_df.sort_values(by='date_str', ascending=True)
# Select only top topics:
k=20
top_k_topics = list(topic_gb.sort_values(by='count', ascending=False)[:k]['Topic'])

# Add 1 to topic number (like before):
named_top_k_topics = list(np.array(top_k_topics)+1)

# Filter df by topics:
cols = []
for topic in top_k_topics:
    cols.append("Topic " + str(topic))
    
cols.append("date_x")
cols.append('covid cases')
daily_top_topic_dist_df = daily_topic_dist_df[cols]
cols.remove("date_x")
cols.remove("covid cases")
# Trend of Topics over Time:

# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

for topic in cols:
    
    fig.add_trace(
      go.Scatter(x=daily_top_topic_dist_df['date_x'], y=savgol_filter(daily_top_topic_dist_df[topic],11,3), name=topic),
      secondary_y=False,)

# Add traces
fig.add_trace(
      go.Scatter(x=daily_top_topic_dist_df['date_x'], y=savgol_filter(daily_top_topic_dist_df['covid cases'],11,3), name="Covid Cases"),
      secondary_y=True,
)


# Set x-axis title
fig.update_xaxes(title_text="Date")
# Set y-axes titles
fig.update_yaxes(title_text="<b>Topic Distribution</b>", secondary_y=False)
fig.update_yaxes(title_text="<b>Covid Cases</b>", secondary_y=True)
fig.show()

# Terms of top topics:
print(topics[topics['Topic'].isin(top_k_topics)])
