"""
In this section we trained a model on 32 million tweets after the entire filtering and cleaning process.
The model is trained for a random corpus and in addition is also trained on Corona tweets, news feeds, capital market tweets and capital market news.
The data was processed using a package called turicreate designed to work with large data.
It should be noted that after each step in the process - we saved the information because the process is very heavy and may crash.
A computer with a very powerful processor is required.
We used Amazon aws with 64GB RAM and 52 CPU Machine

The code was originally written in AWS and we changed it so that it can run on any machine - if the database does not exceed 10M should not be a problem

More Information about the word2vec model can be found on Genesim's website: https://radimrehurek.com/gensim/models/word2vec.html
"""
from utils import *
from preprocess import *
from gensim.test.utils import get_tmpfile
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models import Word2Vec

class EpochLogger(CallbackAny2Vec):
     '''Callback to log information about training'''

     def __init__(self):
         self.epoch = 0
     def on_epoch_begin(self, model):
         print("Epoch #{} start".format(self.epoch))
     def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        print('Loss after epoch {}: {}'.format(self.epoch, loss))
        self.epoch += 1

epoch_logger = EpochLogger()
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
sf['clean_text'] = sf['text'].apply(lambda x: preprocess_tweet(tweet=x, min_text_len=min_text_len, min_word_len=min_word_len))                                                                      
sf = sf[['clean_text']]
sf = sf.dropna()
sf.materialize()

#tokens
sf['tokens'] = sf['clean_text'].apply(lambda x: x.split())
sf = sf[['tokens']]
sf.materialize()

#build wor2Vec Model
tokens = sf['tokens']
w2v_model = Word2Vec(min_count=20,
                     window=2,
                     size=300,
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=20,
                     workers=32)
w2v_model.build_vocab(tokens, progress_per=10000)
# train - could take long time
w2v_model.train(tokens, total_examples=w2v_model.corpus_count, epochs=25, report_delay=1,callbacks=[epoch_logger])

# keep model trainable
w2v_model.init_sims(replace=True)

#save the model
model_path = '/content/drive/My Drive/big_data/models/word2vec/model_w2v.wv'
w2v_model.save(model_path)





