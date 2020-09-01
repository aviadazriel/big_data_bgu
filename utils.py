import warnings
import pandas as pd
pd.set_option("display.max_colwidth", 200) 
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import os
import emoji
import spacy
from pylab import rcParams
from scipy.signal import savgol_filter
import _pickle as cPickle
from sklearn.decomposition import PCA
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import copy
import datetime
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from collections import Counter
import pandas as pd
from tqdm import tqdm
from wordcloud import WordCloud
from dateutil import parser
from nltk.stem import WordNetLemmatizer
from plotly.offline import iplot
import seaborn as sns
import re
import nltk
import string
from wordcloud import WordCloud
from PIL import Image
import codecs
import cv2
import numpy as np
from string import punctuation 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from sklearn.decomposition import LatentDirichletAllocation
import plotly.express as px
from textblob import TextBlob
import pyLDAvis
import turicreate as tc
import pyLDAvis.gensim
import matplotlib.pyplot as plt
# %matplotlib inline
tqdm.pandas()
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def word_vector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in tokens:
        try:
            vec += model_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec

def save_pickle_object(obj, obj_name, url):
    file = open(url + obj_name + '.obj', 'wb')
    cPickle.dump(obj, file)
    print('Info: saved {o} pickle file.'.format(o=str(obj_name)))

def load_pickle_object(obj_name, url):
    file = open(url + obj_name + '.obj', 'rb')
    obj = cPickle.load(file)
    print('Info: uploading {o} pickle file.'.format(o=str(obj_name)))
    return obj

def parse_date(x):
  
  try:
    date = parser.parse(x)
  except:
    date = np.nan

  return date


def concat_dfs_by_row(list_of_dfs):
    new_list_Of_dfs = []

    if len(list_of_dfs) < 2:
        print('Error: less than two DFs.')
        return None

    for df in tqdm(list_of_dfs):
        df.reset_index(drop=True, inplace=True)
        new_list_Of_dfs.append(df)

    all_df = pd.concat(new_list_Of_dfs, axis=0, ignore_index=True)

    return all_df

def concat_dfs_by_col(list_of_dfs):
    new_list_Of_dfs = []

    if len(list_of_dfs) < 2:
        print('Error: less than two DFs.')
        return None

    row_shape = list_of_dfs[0].shape[0]
    for df in list_of_dfs:
        assert df.shape[0] == row_shape
        df.reset_index(drop=True, inplace=True)
        new_list_Of_dfs.append(df)

    all_df = pd.concat(new_list_Of_dfs, axis=1)

    return all_df

def plot_histogram(df, column, title):
    ax = sns.distplot(df[column])
    plt.suptitle(title)
    plt.show()


def get_sentiment(text):
   return  TextBlob(text).sentiment.polarity


def word_cloud(df , column, max_words=200,pic_path = None):
  all_words = df[column].dropna()
  words = " ".join(" ".join(list(all_words)).split())
  plt.subplots(figsize=(28,12))

  if pic_path is  not None:
    mask = cv2.imread(pic_path, 1)
    mask2 = np.array(Image.open(pic_path)) 
    wordcloud = WordCloud(background_color="white",#mode="RGBA",
                                max_words=max_words,mask=mask,max_font_size=256#,width=mask.shape[1],height=mask.shape[0]
                                ).generate(words)
  else:  
    wordcloud = WordCloud(background_color="white", mode="RGBA",
                              width=2048,
                              max_words=max_words,
                              height=1024,
                              collocations=False).generate(words)

  # show
  plt.imshow(wordcloud, interpolation="bilinear")
  plt.axis("off")
  plt.figure()
  if pic_path is  not None:
    plt.imshow(mask2, cmap=plt.cm.gray, interpolation='bilinear')
    plt.axis("off")
    plt.show()

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

def create_pie_chart(df, group_by_col, count_col, title):
  gb = df.groupby(by=group_by_col).count()
  gb['prec'] = gb[count_col].apply(lambda x: round((x/sum(gb[count_col]))*100,2))
  group_size = gb[count_col]
  group_labels = ['Other Tweets ' + str(gb['prec'][0]) + "%",'Covid Tweets ' + str(gb['prec'][1]) + "%"]
  custom_colors = ['skyblue','tomato']
  plt.figure(figsize = (5,5))
  plt.pie(group_size, labels = group_labels, colors = custom_colors)
  central_circle = plt.Circle((0,0), 0.5, color = 'white')
  fig = plt.gcf()
  fig.gca().add_artist(central_circle)
  plt.rc('font', size = 12)
  plt.title(title, fontsize = 15)
  plt.show()


def compute_coherence_values(dictionary, corpus, texts, limit, random_state, update_every, passes, alpha, per_word_topics, chunksize, start=2, step=3):
    import warnings
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    coherence_values = []
    model_list = []
    for num_topics in tqdm(range(start, limit, step)):

        model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=random_state,
                                                update_every=update_every, chunksize=chunksize, passes=passes, alpha=alpha,
                                                per_word_topics=per_word_topics)

        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values
