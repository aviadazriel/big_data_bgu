def load_data(input_url, filter_word=None, remove_leader_tweet=False, random_sample = None):
  # Get file names:
  file_list = []

  for root, dirs, files in os.walk(input_url):
    for file in files:
      file_list.append(os.path.join(root,file))

  # Filters files:
  if filter_word:
    file_list = [file for file in file_list if filter_word in file]

  # Remove files:
  if remove_leader_tweet:
    file_list = [file for file in file_list if 'trump_tweets' not in file]

  # Load files:
  dfs = []
  for file in tqdm(file_list):
    if file.split('/')[-1]=="2020_06_6.csv":
      df = pd.read_csv(file, sep='\t', engine='python')
    else:
      try:
        df = pd.read_csv(file, low_memory=False)
      except:
        print(file)
        df = pd.read_csv(file, engine='python')

    if random_sample is not None:
      df = df.sample(frac=random_sample)

    dfs.append(df)
    
  # Merge data:
  df = concat_dfs_by_row(list_of_dfs=dfs)
  print("\nDF shape is {s}".format(s=str(df.shape[0])))

  return df

def lower_text(text):
    text = text.lower()
    return text

def is_relevent_tweet(text,words_list):

  if text==str(np.nan):
    return False
  text = lower_text(text=text)
  for word in words_list:
    if word.lower() in text:
      return True
  return False

def preprocess_tweet(tweet, min_text_len=2, min_word_len=3, keywords_list=None, keywords_replacement_word=None, stem=True, remove_hashtag=True):

  tweet = tweet.lower()
  tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '', tweet) 
  tweet = re.sub('@[^\s]+', '', tweet) 
  tweet = re.sub('[!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@]+', ' ', tweet)
  tweet = re.sub('([0-9]+)', '', tweet)
  
  if remove_hashtag:
    tweet = re.sub('#[^\s]+', '', tweet) 
  else:
    tweet = tweet.replace('#', '')

  tweet = tweet.replace('amp', '')

  if (keywords_list is not None) and (keywords_replacement_word is not None):
    for key in keywords_list:
        if key.replace('#','') in tweet:    
          tweet = tweet.replace(key.replace('#','') , ' ' + str(keywords_replacement_word) + ' ')

  punctuation_list = list(string.punctuation)
  punctuation_list.append('…')
  for char in punctuation_list:
      if char in tweet:
          tweet = tweet.replace(char, '')

  tweet = re.sub('\s+', ' ', tweet)

  allchars = [str for str in tweet]
  emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]
  tweet = ' '.join([str for str in tweet.split() if not any(i in str for i in emoji_list)])
  tweet = re.sub(r"[^A-Za-z]+", ' ', tweet)  
  tokens = word_tokenize(tweet) 

  stop_words = stopwords.words('english')
  stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
  tokens = [word for word in tokens if word not in stop_words]
  
  if stem:
    word_rooter = nltk.stem.snowball.PorterStemmer(ignore_stopwords=False).stem
    tokens = [word_rooter(word) if '#' not in word else word
                          for word in tokens]
  tokens = [word for word in tokens if len(word)>=min_word_len]

  if len(tokens)<min_text_len:
    return np.nan

  clean_tweet = ' '.join(tokens)
  
  return clean_tweet

def write_data(df, url):
    df.to_csv(url, index=False)
    print('Saved csv in {u}'.format(u=str(url)))

def reco_device(tweet):
  try:
    tweet = re.findall(r'>(.+?)<', tweet)[0]
    if 'iPhone' in tweet or ('iOS' in tweet):
        return 'iPhone'
    elif 'Android' in tweet:
        return 'Android'
    elif 'Mobile' in tweet or ('App' in tweet):
        return 'Mobile device'
    elif 'Mac' in tweet:
        return 'Mac'
    elif 'Windows' in tweet:
        return 'Windows'
    elif 'Bot' in tweet:
        return 'Bot'
    elif 'Web' in tweet:
        return 'Web'
    elif 'Instagram' in tweet:
        return 'Instagram'
    elif 'Blackberry' in tweet:
        return 'Blackberry'
    elif 'iPad' in tweet:
        return 'iPad'
    elif 'Foursquare' in tweet:
        return 'Foursquare'
    else:
        return '-'
  except:
    return None

def convertDate(d):
    try:
      new_date = datetime.datetime.strptime(d,"%d/%m/%Y %H:%M")
    except:
      new_date = datetime.datetime.strptime(d,"%Y-%m-%d %H:%M:%S")
    return new_date.date()

def preprocessing(df, random_sample = None, specific_columns = None):
  print('Start Preprocessing')
  # Filter cols:
  if specific_columns is not None:
    df = df[specific_columns]
  print('filter unknown sources')
  shape = len(df)
  df['source_clean'] = df['source'].apply(lambda x: reco_device(x))
  relevant_device = ['Web', 'Android','Mobile device', None,'iPad','iPhone']
  df = df[df['source_clean'].isin(relevant_device)]
  print(f'remove rows: {shape-len(df)}' )
  
  print('Parse Date')
  df['created_at'] = df['created_at'].apply(lambda x: parse_date(x))
  df['date'] = df['created_at'].apply(lambda x: x.date())
  
  # Remove nan values:
  df = df.dropna(subset=['created_at','id','text'])

  if random_sample is not None:
    df = df.sample(frac=random_sample)

  # Remove duplicates:
  print('Remove duplicates')
  df = df.drop_duplicates()
  df = df.sort_values('created_at') 

  # Convert dtypes:
  for col_name in ['text','State_abbv']:
    df[col_name] = df[col_name].astype(str)

  df = df.reset_index(drop=True)

  print("After pre-proccessing DF shape is {s}".format(s=str(df.shape[0])))
  return df
