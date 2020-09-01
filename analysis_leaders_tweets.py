from utils import *
from pylab import rcParams

def get_subjectivity(text):
  return  TextBlob(text).sentiment.subjectivity
  
def sentiment_recognize(score):
  if score > 0:
    return 'Positive'
  elif score < 0:
    return 'Negetive'
  else:
    return 'Netural'


def find_status_id(id, id_list):

  if id==np.nan:
    return np.nan

  id = str(id)[:16].replace('.','')

  for status_id in id_list:
    if id in str(status_id):
      return str(status_id)
  return np.nan

# MAIN 

trump_tweets_path = "drive/My Drive/Colab Notebooks/Big Data/final_project/Data/BGU DATA/Leaders Tweets and Retweets/trump_tweets.csv"
# Load trump tweets:
trump_df = pd.read_csv(trump_tweets_path)

# Data preprocessing:
trump_df = trump_df.rename(columns={'date': 'created_at'})
trump_df['created_at'] = trump_df['created_at'].apply(lambda x: convertDate(str(x)))
trump_df = trump_df.sort_values("created_at")
trump_df = trump_df.dropna(subset=['created_at','id','text'])
trump_df['status_id'] = trump_df['permalink'].apply(lambda x: x.split('/')[-1])

# Tag corona tweets:
trump_df['covid_tweet'] = trump_df['text'].apply(lambda x: is_relevent_tweet(text=x,words_list=corona_keyword))

# Clean text:
min_text_len = 1
min_word_len = 3
trump_df['clean_text'] = trump_df['text'].progress_apply(lambda x: preprocess_tweet(tweet=x, min_text_len=min_text_len, min_word_len=min_word_len, stem=True))
trump_df['clean_text_no_stem'] = trump_df['text'].progress_apply(lambda x: preprocess_tweet(tweet=x, min_text_len=min_text_len, min_word_len=min_word_len, stem=False))
trump_df = trump_df[~trump_df['clean_text'].isna()]

# Extract sentiment:
trump_df['sentiment'] = trump_df['clean_text'].apply(lambda x: get_sentiment(x))  
trump_df['sentiment_desc'] = trump_df['sentiment'].apply(lambda x: sentiment_recognize(x))

# Extract subjectivity:
trump_df['subjectivity'] = trump_df['clean_text'].apply(lambda t: get_subjectivity(t))

# Create pie of trump tweet sentiment:
counts = Counter(trump_df['sentiment_desc'])
fig = px.pie( values=[float(v) for v in counts.values()], names=[k for k in counts]
                                  ,title='Trump\'s tweets sentiment description')
fig.show()

# Word cloud of Trump's tweets:
column = "clean_text"
pic = "./trump_pic.jpg" 
rcParams['figure.figsize'] = 5, 5
word_cloud(df=trump_df, column=column, max_words = 2000, pic_path = pic)

"""Trump Retweets"""

# Load trump retweets:
trump_retweet_path = "trump_retweets.csv"
df = pd.read_csv(trump_retweet_path, low_memory=False)

# Data Preprocessing:
specific_columns = ['created_at','id','source','text','in_reply_to_status_id','in_reply_to_user_id','user.id','State_abbv']
preprocessed_df = preprocessing(df=df, specific_columns = specific_columns)

# Clean text:
min_text_len = 1
preprocessed_df['clean_text'] = preprocessed_df['text'].apply(lambda x: preprocess_tweet(tweet=x, min_text_len=min_text_len))
preprocessed_df = preprocessed_df[~preprocessed_df['clean_text'].isna()]

# Extract sentiment:
preprocessed_df['sentiment'] = preprocessed_df['clean_text'].apply(lambda x: get_sentiment(x))

"""Analysis between Trump's Tweets and the replies tweets"""

# Filter for trump retweets and replies:
trump_user_id = 25073877
trump_replies_df = preprocessed_df[preprocessed_df['in_reply_to_user_id']==f'{trump_user_id}.0'].reset_index(drop=True)
trump_replies_df['date'] = trump_replies_df['date'].apply(lambda d: d.date())
trump_replies_df_agg = trump_replies_df.groupby(['date']).mean().reset_index()

# Filter per dates:
min_date = min(trump_df['created_at'])
trump_replies_df = trump_replies_df[trump_replies_df['date']>=min_date]

# Create sentiment over time graph:
id_list = list(set(trump_df['status_id']))
trump_replies_df['status_id'] = trump_replies_df['in_reply_to_status_id'].apply(lambda x: find_status_id(id=x, id_list=id_list))

print(f'After filtering the match between Trump\'s tweet and his reactions, {len(trump_replies_df[~trump_replies_df["status_id"].isna()])} lines were found')

trump_replies_sentiment_gb = trump_replies_df[['status_id','sentiment']].groupby(by='status_id').mean().reset_index()
trump_replies_sentiment_gb = trump_replies_sentiment_gb.rename(columns={'sentiment':'mean_replies_sentiment'})
trump_replies_sentiment_gb = trump_replies_sentiment_gb.dropna()

# Merge with trump tweets:
new_trump_replies_df = trump_df.merge(trump_replies_sentiment_gb, on='status_id', how='left')
assert new_trump_replies_df['mean_replies_sentiment'].dropna().shape[0]==trump_replies_sentiment_gb.shape[0]
new_trump_replies_df = new_trump_replies_df[['created_at','status_id','covid_tweet','sentiment','subjectivity','mean_replies_sentiment', 'replies']]

covid_spread_state_gb = covid_spread_state_df.groupby(by='date').sum().reset_index()
new_trump_replies_df['date_str'] = new_trump_replies_df['created_at'].astype(str)

new_trump_replies_df = new_trump_replies_df.merge(covid_spread_state_gb, left_on='date_str', right_on='date', how='left')

new_trump_replies_df['covid cases'] = new_trump_replies_df['covid cases'].fillna(value=0)
new_trump_replies_df['covid deaths'] = new_trump_replies_df['covid deaths'].fillna(value=0)

"""Graph Sentiment and Covid Cases"""

all_data_df = new_trump_replies_df.groupby(by='created_at')[['sentiment','mean_replies_sentiment','subjectivity','covid cases','covid deaths', 'replies']].mean().reset_index()

# Create some mock data
t = all_data_df['created_at']
cases = all_data_df['covid cases']
avg = all_data_df['mean_replies_sentiment'].dropna().mean()
comment_sentiment = all_data_df['mean_replies_sentiment'].fillna(avg)
trump_sentiment = all_data_df['sentiment'] 
# subjectivity = all_data_df['subjectivity'] 

# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Scatter(x=t, y=savgol_filter(comment_sentiment,21,5), name="trump comments sentiment", line=dict(color='royalblue', width=2, dash='dot') ),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=t, y=savgol_filter(trump_sentiment,51,3), name="trump tweets sentiment", line=dict(color='green', width=2, dash='dot')),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=t, y=savgol_filter(cases,11,3), name="cases", line=dict(color='red', width=2)),
    secondary_y=True,
)

# Add figure title
fig.update_layout(
    title_text="Trump Sentiment VS COVID-19 Cases"
)
# Set x-axis title
fig.update_xaxes(title_text="Day")
# Set y-axes titles
fig.update_yaxes(title_text="<b>Trump Score</b>", secondary_y=False)
fig.update_yaxes(title_text="<b>Cases</b>", secondary_y=True)
fig.show()

"""HeatMap"""

#General Corr HeatMap
rcParams['figure.figsize'] = 10, 10
a = all_data_df[['sentiment', 'mean_replies_sentiment', 'subjectivity','covid cases','covid deaths','replies' ]].corr()
sns.heatmap(a, xticklabels=a.columns, yticklabels=a.columns, annot=True)  
plt.show()

"""Covid Time Corr HeatMap"""

covid_time_trump_agg = all_data_df[all_data_df['created_at']>pd.to_datetime('20200201',format='%Y%m%d')]
a = covid_time_trump_agg[['sentiment', 'mean_replies_sentiment', 'subjectivity','covid cases','covid deaths','replies']].corr()
sns.heatmap(a, xticklabels=a.columns, yticklabels=a.columns, annot=True)  
plt.show()

"""Biden tweets"""

# Load Biden tweets:
biden_path ='./drive/My Drive/Colab Notebooks/Big Data/final_project/Data/other/twitts (1)/JoeBiden (1).csv'
biden_df = pd.read_csv(biden_path)

# Data preprocessing:
biden_df = biden_df.rename(columns={'date': 'created_at'})
biden_df['created_at'] = biden_df['created_at'].apply(lambda x: convertDate(str(x)))
biden_df = biden_df.sort_values("created_at")
biden_df = biden_df.dropna(subset=['created_at','id','text'])
biden_df['status_id'] = biden_df['permalink'].apply(lambda x: x.split('/')[-1])

# Tag corona tweets:
biden_df['covid_tweet'] = biden_df['text'].apply(lambda x: is_relevent_tweet(text=x,words_list=corona_keyword))

# Clean text:
min_text_len = 1
min_word_len = 3
biden_df['clean_text'] = biden_df['text'].progress_apply(lambda x: preprocess_tweet(tweet=x, min_text_len=min_text_len, min_word_len=min_word_len, stem=True))
biden_df['clean_text_no_stem'] = biden_df['text'].progress_apply(lambda x: preprocess_tweet(tweet=x, min_text_len=min_text_len, min_word_len=min_word_len, stem=False))
biden_df = biden_df[~biden_df['clean_text'].isna()]

# Extract sentiment:
biden_df['sentiment'] = biden_df['clean_text'].apply(lambda x: get_sentiment(x))  
biden_df['sentiment_desc'] = biden_df['sentiment'].apply(lambda x: sentiment_recognize(x))

biden_df['subjectivity'] = biden_df['clean_text'].apply(lambda t: get_subjectivity(t))

# Create pie of trump tweet sentiment:
counts = Counter(biden_df['sentiment_desc'])
fig = px.pie( values=[float(v) for v in counts.values()], names=[k for k in counts]
                                  ,title='biden\'s tweets sentiment description')
fig.show()

# Word cloud of Trump's tweets:
column = "clean_text"
pic = "./asdf.png" 
rcParams['figure.figsize'] = 5, 5
word_cloud(df=biden_df, column=column, max_words = 4000, pic_path = pic)

"""###Biden Retweets"""

# Load Biden retweets:
biden_retweets_path = "/content/drive/My Drive/big_data/tables/biden_retweets.csv"
df = pd.read_csv(biden_retweets_path, low_memory=False) 

# Data Preprocessing:
specific_columns = ['created_at','id','source','text','in_reply_to_status_id','in_reply_to_user_id','user.id','State_abbv']
preprocessed_df = preprocessing(df=df, specific_columns = specific_columns)

# Clean text:
min_text_len = 1
preprocessed_df['clean_text'] = preprocessed_df['text'].apply(lambda x: preprocess_tweet(tweet=x, min_text_len=min_text_len))

preprocessed_df = preprocessed_df[~preprocessed_df['clean_text'].isna()]

# Extract sentiment:
preprocessed_df['sentiment'] = preprocessed_df['clean_text'].apply(lambda x: get_sentiment(x))

"""Analysis between Biden's Tweets and the replies tweets"""

# Filter for trump retweets and replies:
biden_user_id = 939091
biden_replies_df = preprocessed_df[preprocessed_df['in_reply_to_user_id']==f'{biden_user_id}.0'].reset_index(drop=True)
# biden_replies_df['date'] = biden_replies_df['date'].apply(lambda d: d.date())
biden_replies_df_agg = biden_replies_df.groupby(['date']).mean().reset_index()

biden_replies_df_agg.head(5)

# Filter per dates:
min_date = min(biden_df['created_at'])
biden_replies_df = biden_replies_df[biden_replies_df['date']>=min_date]

# Create sentiment over time graph:
id_list = list(set(biden_df['status_id']))
biden_replies_df['status_id'] = biden_replies_df['in_reply_to_status_id'].apply(lambda x: find_status_id(id=x, id_list=id_list))

print(f'After filtering the match between Biden\'s tweet and his reactions, {len(biden_replies_df[~biden_replies_df["status_id"].isna()])} lines were found')

biden_replies_sentiment_gb = biden_replies_df[['status_id','sentiment']].groupby(by='status_id').mean().reset_index()
biden_replies_sentiment_gb = biden_replies_sentiment_gb.rename(columns={'sentiment':'mean_replies_sentiment'})
biden_replies_sentiment_gb = biden_replies_sentiment_gb.dropna()

# Merge with trump tweets:
new_biden_replies_df = biden_df.merge(biden_replies_sentiment_gb, on='status_id', how='left')
assert new_biden_replies_df['mean_replies_sentiment'].dropna().shape[0]==biden_replies_sentiment_gb.shape[0]

new_biden_replies_df = new_biden_replies_df[['created_at','status_id','covid_tweet','sentiment','subjectivity','mean_replies_sentiment', 'replies']]

covid_spread_state_gb = covid_spread_state_df.groupby(by='date').sum().reset_index()
new_biden_replies_df['date_str'] = new_biden_replies_df['created_at'].astype(str)

new_biden_replies_df = new_biden_replies_df.merge(covid_spread_state_gb, left_on='date_str', right_on='date', how='left')

new_biden_replies_df['covid cases'] = new_biden_replies_df['covid cases'].fillna(value=0)
new_biden_replies_df['covid deaths'] = new_biden_replies_df['covid deaths'].fillna(value=0)

"""Graph** sentiments and Covid Cases"""

all_data_biden_df = new_biden_replies_df.groupby(by='created_at')[['sentiment','mean_replies_sentiment','subjectivity','covid cases','covid deaths', 'replies']].mean().reset_index()

# all_data_df['mean_replies_sentiment'].dropna()

# Create some mock data
t = all_data_biden_df['created_at']
cases = all_data_biden_df['covid cases']
avg = all_data_biden_df['mean_replies_sentiment'].dropna().mean()
comment_sentiment = all_data_biden_df['mean_replies_sentiment'].fillna(avg)
biden_sentiment = all_data_biden_df['sentiment'] 

# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Scatter(x=t, y=savgol_filter(comment_sentiment,21,5), name="Biden comments sentiment", line=dict(color='royalblue', width=2, dash='dot') ),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=t, y=savgol_filter(biden_sentiment,51,3), name="Biden tweets sentiment", line=dict(color='green', width=2, dash='dot')),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=t, y=savgol_filter(cases,11,3), name="cases", line=dict(color='red', width=2)),
    secondary_y=True,
)

# Add figure title
fig.update_layout(
    title_text="Biden Sentiment VS COVID-19 Cases"
)
# Set x-axis title
fig.update_xaxes(title_text="Day")
# Set y-axes titles
fig.update_yaxes(title_text="<b>Biden Score</b>", secondary_y=False)
fig.update_yaxes(title_text="<b>Cases</b>", secondary_y=True)
fig.show()

"""HeatMap
Sentiment | Mean Replies Sentiment | Subjectivity | Covid Cases | Covid Deaths Replies
"""

#General Corr HeatMap

rcParams['figure.figsize'] = 10, 10
a = all_data_biden_df[['sentiment', 'mean_replies_sentiment', 'subjectivity','covid cases','covid deaths','replies' ]].corr()
sns.heatmap(a, xticklabels=a.columns, yticklabels=a.columns, annot=True)  
plt.show()

"""#####Covid Time Corr HeatMap"""

covid_time_biden_agg = all_data_biden_df[all_data_biden_df['created_at']>pd.to_datetime('20200201',format='%Y%m%d')]
a = covid_time_biden_agg[['sentiment', 'mean_replies_sentiment', 'subjectivity','covid cases','covid deaths','replies']].corr()
sns.heatmap(a, xticklabels=a.columns, yticklabels=a.columns, annot=True)  
plt.show()
