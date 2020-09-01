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
