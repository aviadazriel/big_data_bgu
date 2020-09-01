from utils import *
def load_stock_prices(sector, dict_stock_sector, url):

  stocks = dict_stock_sector[sector]

  dfs = []
  for stock in stocks:
    
    df = pd.read_csv(url+str(stock)+".csv")
    df = df.drop(columns=['Unnamed: 0'])
    df  = df.dropna()
    df['Date'] = df['Date'].apply(lambda x: pd.to_datetime(x,format= '%Y-%m-%d').date())
    df['stock'] = stock
    df['sector'] = sector
    dfs.append(df)

  if len(dfs)==1:
    return dfs[0]

  stock_df = concat_dfs_by_row(list_of_dfs=dfs)
  return stock_df

def plot_stock_sentiment(stock_gb):
  t = clean_energy_stock['date']
  sentiment = stock_gb['sentiment']
  price = stock_gb['Close']

  fig = make_subplots(specs=[[{"secondary_y": True}]])

  fig.add_trace(
      go.Scatter(x=t, y=savgol_filter(sentiment,21,3), name="Sentiment", line=dict(color='red', width=2) ),
      secondary_y=False,
  )

  fig.add_trace(
      go.Scatter(x=t, y=price, name="Price", line=dict(color='royalblue', width=2)),
      secondary_y=True,
  )

  fig.update_layout(
      title_text="{s} stock daily price and sentiment during covid-19".format(s=str(stock))
  )
  fig.update_xaxes(title_text="Day")
  fig.update_yaxes(title_text="<b>Sentiment</b>", secondary_y=False)
  fig.update_yaxes(title_text="<b>Price</b>", secondary_y=True)
  fig.show()

def plot_stock(df, stock):

  d = df[df['stock']==stock]
  t = d["Date"]
  price = d["Close"]
  volume = d["Volume"]

  # Create figure with secondary y-axis
  fig = make_subplots(specs=[[{"secondary_y": True}]])

  # Add traces
  fig.add_trace(
      go.Scatter(x=t, y=price, name="Stock Price"),
      secondary_y=False,
  )
  fig.add_trace(
      go.Scatter(x=t, y=volume, name="Volume"),
      secondary_y=True,
  )
  # Add figure title
  fig.update_layout(
      title_text=f'{stock}: Stock Price and Volume'
  )
  # Set x-axis title
  fig.update_xaxes(title_text="Date")
  # Set y-axes titles
  fig.update_yaxes(title_text="<b>Stock Price</b>", secondary_y=False)
  fig.update_yaxes(title_text="<b>Volume</b>", secondary_y=True)
  fig.show()

def recognize_sector(txt, market_relavant_words, clean_energy_relevant_words, Crude_relevant_words, Solar_Energy_relevant_words, Gasoline_relevant_word):
  txt = txt.lower()

  for word in ['tsla','tesla','elon_mask','elon mask','elonmask']:
    if word in txt:
      return 'tesla'

  for word in Gasoline_relevant_word:
    if word in txt:
      return 'Gasoline'
  
  for word in Solar_Energy_relevant_words:
    if word in txt:
      return 'Solar Energy'

  for word in Crude_relevant_words:
    if word in txt:
      return 'Crude'
  
  for word in clean_energy_relevant_words:
    if word in txt:
      return 'Clean Energy'
  
  for word in market_relavant_words:
    if word in txt:
      return 'stocks'

  if 'russian' in txt:
    return 'russian'

  if 'saudi arabia' in txt:
    return 'saudi'
  return 'unknown'

stock_prices_path =  '/content/drive/My Drive/big_data/tables/stocks_prices.csv'
stock_prices_df = pd.read_csv(stock_prices_path, index_col='Unnamed: 0')
stock_prices_df['Date'] = stock_prices_df['Date'].apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d'))
stock_prices_df

# Load data:
input_url = "/content/drive/My Drive/big_data/tables/stock_tweets.csv"
stocks_df = pd.read_csv(input_url, low_memory=False)
print(stocks_df.shape)

# Config:
market_relavant_words = ['s&p','nasda','vix','stock market','energy stock', 'es_f','sp5', 'S&amp;P' 'ixic']
clean_energy_relevant_words = ['nee','nextera','tsla', 'tesla','pbw','invesco wilderhill','clean energy','cleanenergy','clean_energy', 'elon musk', 'elonmask', 'elon_mask']
Crude_relevant_words = ['uco', 'bno','clf','cl=f','cl_f','black oil','blackoil','black_oil','crude','wti', 'brent','dbo','oott', 'oil trading', 'oil_trading','oiltrading','fintwit','oil price','oilprice','oil_price','usa oil', 'usoil', 'usaoil']
Solar_Energy_relevant_words = ['tan','solar']
Gasoline_relevant_word = ['gasoline','gas','uga']
dict_stock_sector  = {'Solar Energy': ['TAN'], 'Clean Energy': ['NEE', 'TSLA', 'PBW'], 'Crude': ['UCO','BNO','CL=F'], 'Gasoline': ['UGA']}

stocks_df['text'] = stocks_df['text'].astype(str)
stocks_df['source'] = stocks_df['source'].astype(str)
stocks_df['created_at'] = stocks_df['created_at'].astype(str)

# Data preprocessing:
stocks_df = stocks_df[~stocks_df['text'].str.lower().str.contains("toilet")]
cols = ['created_at','id','source', 'text','in_reply_to_status_id','in_reply_to_user_id','user.id','State_abbv']
stocks_df = preprocessing(stocks_df, specific_columns = cols)
stocks_df['sector'] = stocks_df['text'].progress_apply(lambda x: recognize_sector(x, market_relavant_words, clean_energy_relevant_words, Crude_relevant_words, Solar_Energy_relevant_words, Gasoline_relevant_word)) 
stocks_df = stocks_df[stocks_df['sector'].isin(['Clean Energy','Crude','Gasoline','Solar Energy'])]

print(f'number of tweets that found: {stocks_df.shape[0]}')

# Sector tweets distribution:
stocks_gb = stocks_df.groupby('sector').count()['id'].reset_index()
sns.barplot(stocks_gb['sector'],stocks_gb['id'])
sns.set(rc={'figure.figsize':(10,10)})
plt.xlabel("Sector")
plt.ylabel("Frequency")
plt.title("Sectors tweets distribution")
plt.show()

"""Clean Energy"""

# Filter tweets per sector:
clean_energy_df = stocks_df[stocks_df['sector']=='Clean Energy']

# Load data:
clean_energy_prices = stock_prices_df[stock_prices_df['sector']=='Clean Energy']

# Plot stock prices:
for stock in dict_stock_sector['Clean Energy']:
  plot_stock(df=clean_energy_prices, stock=stock)

# Clean text:
min_text_len = 1
min_word_len = 3
clean_energy_df['clean_text'] = clean_energy_df['text'].progress_apply(lambda x: preprocess_tweet(tweet=x, min_text_len=min_text_len,
                                                                                                  min_word_len=min_word_len, remove_hashtag=True))
clean_energy_df = clean_energy_df[~clean_energy_df['clean_text'].isna()]

# Extract sentiment:
clean_energy_df['sentiment'] = clean_energy_df['clean_text'].progress_apply(lambda x: get_sentiment(x))

# Create word cloud:
word_cloud(clean_energy_df, column='clean_text', max_words = 1000)

"""We chose to present the PBW ETF because it best represents the sector versus the sentiment of the tweets"""

# Plot stock prices and daily sector sentiment:
stock = 'PBW'
clean_energy_prices_stock = clean_energy_prices[clean_energy_prices['stock']==stock]
clean_energy_gb = clean_energy_df.groupby(by=['date']).mean()['sentiment'].reset_index()
clean_energy_prices_gb = clean_energy_prices_stock.groupby(by=['Date']).mean()[['Open','High','Low','Close']].reset_index()
clean_energy_prices_gb['Date'] = clean_energy_prices_gb['Date'].astype(str)
clean_energy_gb['date'] = clean_energy_gb['date'].astype(str)
clean_energy_stock = clean_energy_gb.merge(clean_energy_prices_gb, how='inner', right_on='Date', left_on='date')
plot_stock_sentiment(stock_gb=clean_energy_stock)

"""Solar Energy"""

# Filter tweets per sector:
solar_energy_df = stocks_df[stocks_df['sector']=='Solar Energy']

# Load data:
solar_energy_prices = stock_prices_df[stock_prices_df['sector']=='Solar Energy']

# Plot stock prices:
for stock in dict_stock_sector['Solar Energy']:
  plot_stock(df=solar_energy_prices, stock=stock)

# Clean text:
min_text_len = 1
min_word_len = 3
solar_energy_df['clean_text'] = solar_energy_df['text'].progress_apply(lambda x: preprocess_tweet(tweet=x, min_text_len=min_text_len,
                                                                                                  min_word_len=min_word_len, remove_hashtag=True))
solar_energy_df = solar_energy_df[~solar_energy_df['clean_text'].isna()]

# Extract sentiment:
solar_energy_df['sentiment'] = solar_energy_df['clean_text'].progress_apply(lambda x: get_sentiment(x))

# Create word cloud:
word_cloud(solar_energy_df, column='clean_text', max_words = 1000)

# Plot stock prices and daily sector sentiment:
stock = 'TAN'
solar_energy_prices_stock = solar_energy_prices[solar_energy_prices['stock']==stock]
solar_energy_gb = solar_energy_df.groupby(by=['date']).mean()['sentiment'].reset_index()
solar_energy_prices_gb = solar_energy_prices_stock.groupby(by=['Date']).mean()[['Open','High','Low','Close']].reset_index()
solar_energy_gb['date'] = solar_energy_gb['date'].astype(str)
solar_energy_prices_gb['Date'] = solar_energy_prices_gb['Date'].astype(str)
solar_energy_stock = solar_energy_gb.merge(solar_energy_prices_gb, how='inner', right_on='Date', left_on='date')
plot_stock_sentiment(stock_gb=solar_energy_stock)

"""## Crude"""

# Filter tweets per sector:
crude_df = stocks_df[stocks_df['sector']=='Crude']

# Load data:
crude_prices = stock_prices_df[stock_prices_df['sector']=='Crude']

# Plot stock prices:
for stock in dict_stock_sector['Crude']:
  plot_stock(df=crude_prices, stock=stock)

# Clean text:
min_text_len = 1
min_word_len = 3
crude_df['clean_text'] = crude_df['text'].progress_apply(lambda x: preprocess_tweet(tweet=x, min_text_len=min_text_len,
                                                                                                  min_word_len=min_word_len, remove_hashtag=True))
crude_df = crude_df[~crude_df['clean_text'].isna()]

# Extract sentiment:
crude_df['sentiment'] = crude_df['clean_text'].progress_apply(lambda x: get_sentiment(x))

# Create word cloud:
word_cloud(crude_df, column='clean_text', max_words = 1000)

# Plot stock prices and daily sector sentiment:
for stock in dict_stock_sector['Crude']:
  crude_prices_stock = crude_prices[crude_prices['stock']==stock]
  crude_gb = crude_df.groupby(by=['date']).mean()['sentiment'].reset_index()
  crude_prices_gb = crude_prices_stock.groupby(by=['Date']).mean()[['Open','High','Low','Close']].reset_index()

  crude_gb['date'] = crude_gb['date'].astype(str)
  crude_prices_gb['Date'] = crude_prices_gb['Date'].astype(str)


  crude_stock = crude_gb.merge(crude_prices_gb, how='inner', right_on='Date', left_on='date')
  plot_stock_sentiment(stock_gb=crude_stock)

"""Gasoline"""

# Filter tweets per sector:
gasoline_df = stocks_df[stocks_df['sector']=='Gasoline']

# Load data:
gasoline_prices = stock_prices_df[stock_prices_df['sector']=='Gasoline']

# Plot stock prices:
for stock in dict_stock_sector['Gasoline']:
  plot_stock(df=gasoline_prices, stock=stock)

# Clean text:
min_text_len = 1
min_word_len = 3
gasoline_df['clean_text'] = gasoline_df['text'].progress_apply(lambda x: preprocess_tweet(tweet=x, min_text_len=min_text_len,
                                                                                                  min_word_len=min_word_len, remove_hashtag=True))
gasoline_df = gasoline_df[~gasoline_df['clean_text'].isna()]

# Extract sentiment:
gasoline_df['sentiment'] = gasoline_df['clean_text'].progress_apply(lambda x: get_sentiment(x))

# Create word cloud:
word_cloud(gasoline_df, column='clean_text', max_words = 1000)

# Plot stock prices and daily sector sentiment:
stock = 'UGA'
gasoline_prices_stock = gasoline_prices[gasoline_prices['stock']==stock]
gasoline_gb = gasoline_df.groupby(by=['date']).mean()['sentiment'].reset_index()
gasoline_prices_gb = gasoline_prices_stock.groupby(by=['Date']).mean()[['Open','High','Low','Close']].reset_index()
gasoline_gb['date'] = gasoline_gb['date'].astype(str)
gasoline_prices_gb['Date'] = gasoline_prices_gb['Date'].astype(str)
gasoline_stock = gasoline_gb.merge(gasoline_prices_gb, how='inner', right_on='Date', left_on='date')
plot_stock_sentiment(stock_gb=gasoline_stock)
