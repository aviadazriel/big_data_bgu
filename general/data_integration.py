"""
The data were extracted from a variety of different sources, each with a different data structure.
Before starting the process we had to consolidate the data into a common data structure suitable for the analysis of the study.

Once the data is consolidated one can start analyzing and drawing conclusions
"""

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



def load_and_parse_data_to_sframe(input_url, filter_word=None):

  # Get file names:
  file_list = []

  for root, dirs, files in os.walk(input_url):
    for file in files:
      file_list.append(os.path.join(root,file))

  # Filters files:
  if filter_word:
    file_list = [file for file in file_list if filter_word in file]
  file_failed = []
  # Load files:
  sfs = []
  df_shape = 0
  for file in tqdm(file_list):
    try:
      df = pd.read_csv(file, low_memory=False)
    except:
      try:
        df = pd.read_csv(file, engine='python')
      except:
        try:
          df = pd.read_csv(p,sep='\t',encoding="ISO-8859-1")
        except:
          file_failed.append(file)
          print("filed to load file: "+ file)
          continue
    # Preprocessed:
    cols = ['created_at','id','source','text','in_reply_to_status_id','in_reply_to_user_id','user.id','State_abbv']
    df = df[cols]
    df = df.drop_duplicates()
    df = df.dropna(subset=['created_at','id','text'])
    df['in_reply_to_status_id'] = df['in_reply_to_status_id'].astype(str)
    df['in_reply_to_user_id'] = df['in_reply_to_user_id'].astype(str)
    df['user.id'] = df['user.id'].astype(str)
    df['State_abbv'] = df['State_abbv'].astype(str)
    df['id'] = df['id'].astype(str)
    df['text'] = df['text'].astype(str)
    df['created_at'] = df['created_at'].apply(lambda x: parse_date(x))
    df['date'] = df['created_at'].dt.date
    df['hour'] = df['created_at'].dt.hour
    df['date_and_hour'] = df.apply(lambda x: str(x['date'])+" "+str(x['hour']),axis=1)

    # Convert to sframe:
        try:
      sf = tc.SFrame(df)
      df_shape+=df.shape[0]
    except:
      file_failed.append(file)
      print("filed to load file: "+ file)
      continue
      # sf = tc.SFrame(df.replace({pd.np.nan: None}))
    sfs.append(sf)
  failed = []
  # Merge data:
  all_sf = sfs[0]
  for sf in sfs[1:]:
    try:
      all_sf = all_sf.append(sf)
    except:
      print('failed to add some sf')
      failed.append(sf)

  # assert all_sf.shape[0]==df_shape
  print("\nCombine SF shape is {s}".format(s=str(all_sf.shape[0])))

  return all_sf, file_failed, failed


def concat_dfs_by_col(list_of_dfs):
    new_list_Of_dfs = []

    if len(list_of_dfs) < 2:
        print_error('Error: less than two DFs.')
        return None

    row_shape = list_of_dfs[0].shape[0]
    for df in list_of_dfs:
        assert df.shape[0] == row_shape
        df.reset_index(drop=True, inplace=True)
        new_list_Of_dfs.append(df)

    all_df = pd.concat(new_list_Of_dfs, axis=1)

    return all_df
