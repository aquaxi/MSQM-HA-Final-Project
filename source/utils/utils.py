#utility Adapted from class 4
from IPython.display import display, Markdown, HTML
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer 
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
lemmatizer = WordNetLemmatizer()
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.metrics import log_loss, accuracy_score, classification_report, f1_score, roc_auc_score
import nltk
import string
import scikitplot as skplt
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#!/usr/bin/env python
# coding: utf-8

# In[ ]:



def preprocess(text):
    """ Preprocess the text
    
    Steps: (1) lowercase, (2) Lammetize, (3) Remove stop words, (4) Remove punctuations, (5)Remove character with the length size of 1, (6) Tokenization
    
    Args:
        text (str) : the text that require reprocess
    
    Return:  
        processed and tokenized text
    
    """
    stop_words = set(stopwords.words('english'))

    lowered = str.lower(text)
    word_tokens = word_tokenize(lowered)

    words = []
    for w in word_tokens:
        if w not in stop_words: 
            if w not in string.punctuation:
                if w not in ['``',"''",'--']:
                    if len(w) > 1:
                        lemmatized = lemmatizer.lemmatize(w)
                        words.append(lemmatized)    
    return words

class MeanEmbeddingVectorizer(object):
    """ Transform text to embedding using pre-trained word2vec language model
    This function has preloaded word2vec language model and use word2vec to transform texts.
    
    Attributes:
        transform(str) : the word that intend to embed
    """    
    def __init__(self, word2vec):
        self.word2vec = word2vec 
        self.dim = len(word2vec.vectors[0])

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec.vocab]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
    
    def fit_transform(self, X, y=None):
        return self.transform(X)  
    
    
def eval_model(X, y, model, probas = False):
    """ Evaluate the performance of the model
    
    Args:
        X (array): Input variables
        y (array): output variables
        model (estimator): model to be evaluated
        probas (bool): default = False (for most of the machine learning algorithms),  True 
    
    Return:
        Print out: AUC, F1 score, Accuracy, Confusion Matrix
    """
    
    if probas:            
      probas = cross_val_predict(model, X, y, cv=StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42), 
                                  n_jobs=-1, method='predict_proba', verbose=2)
    else: 
      try: 
        probas = model.predict_proba(X)
      except:  #NNT
        probas = model.model.predict(X)

    eval_auc(y, probas)
    
    pred_indices = np.argmax(probas, axis=1)
    classes = np.unique(y)
    preds = classes[pred_indices]

    print('Accuracy: {}\n'.format(accuracy_score(y, preds)))
    skplt.metrics.plot_confusion_matrix(y, preds)
    print(classification_report(y, preds))


from sklearn.metrics import f1_score

def eval_auc(y, probas):
  """ evaluate performance: calculate AUC score 

    return: AUC score (OvR, one versus rest method)
  """
  roc_auc_ovr = []
  classes = list(set(y))
  print('--------')
  for i in range(len(classes)):
    c = classes[i]

    y_auc = y.copy()
    y_auc = [1 if x == c else 0 for x in y_auc]
    y_probas = probas[:,i]
    roc_auc_ovr.append(roc_auc_score(y_auc, y_probas))
    print(f'class {i} AUC OvR: {roc_auc_ovr[i]:.3f}')

  print(f'Avg AUC OvR {np.mean(roc_auc_ovr):.3f}')
  print('--------')
  return roc_auc_ovr


def evaluate_features(X, y, model = LogisticRegression()):
    """ Evaluate the performance of the model
    
    Args:
        X (array): Input variables
        y (array): output variables
        model (estimator): model to be evaluated, default: logistic regression
   
    Return:
        Print out: AUC, F1 score, Accuracy, Confusion Matrix    
    """
    #scaler 
    #scaler = MinMaxScaler()
    #X = scaler.fit_transform(X)
   
    try:           
        probas = cross_val_predict(model, X, y, cv=StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42), 
                                  n_jobs=-1, method='predict_proba', verbose=2)
    except: 
      try:    #models without proba
        probas = model.predict_proba(X)
      except:  #NNT
        probas = model.model.predict(X)

    auc = eval_auc(y, probas)
    
    pred_indices = np.argmax(probas, axis=1)
    classes = np.unique(y)
    preds = classes[pred_indices]

    acc = accuracy_score(y, preds)
    f1 = f1_score(y,preds,average='weighted')
    print(f'Accuracy: {acc:.3f}\n')
    skplt.metrics.plot_confusion_matrix(y, preds)
    print(classification_report(y, preds))
    metrics = {
        'auc':np.mean(auc),
        'accuracy': acc,
        'f1_score':f1
    }
    return metrics
    
    

    
import os
import fasttext
# Prepare document 

def train_fasttext(data, x_col = 'clean', y_col = 'label' ,file = 'fast'):
    """ Train the dataset with Fasttext model 
   
    Args:
        data: dataset in format for dataframe
        x_col : input variables 
        y_col : output variables
        file : name of the file that attempt to save
        
    return: model, performance of model
    """ 
    
    file_name = file + '_'   
    try:
        os.remove(file_name+'train')
        os.remove(file_name+'test')
        print('previous file deleted')
    except:
        print('no exist file')

    for x in ['train','test']:
        with open( file_name + x,'w') as f:
          for i in (data[data['data_type']==x]).index:
            f.write('__label__'+str(data[y_col][i])+' '+data[x_col][i])
            f.write('\n')
        f.close()   
    
    print('Complete loading file, Start training the model')   
    model =fasttext.train_supervised(input= file_name+'train', 
                                   epoch=25, 
                                   wordNgrams=2, 
                                   lr = 0.5)

      #evaluate the result 
    preds = []
    df = data[data['data_type']=='test']

    for i in df.index:
        preds.append(model.predict(df['clean'][i])[0][0][-1])

    print(classification_report(df['label'].astype(str), preds))
    acc = accuracy_score(y, preds)
    f1 = f1_score(y,preds,average='weighted')
    skplt.metrics.plot_confusion_matrix(df['label'].astype(str), preds)
    result = {
    'auc': None,
    'accuracy': acc,
    'f1_score': f1
    }  
    return model, result     

def sanity_check(model):
  """ Check the result of classification  
    
  Arg:
      model (estimator): model to be evaluated
  
  Return:  
      classification count plot 
  """  
  try: 
    test_df['label'] = np.argmax(model.predict(mean_embedding_test), axis=1)
  except:
    test_df['label'] = model.predict(mean_embedding_test)
  
  test_df['label_desc'] =test_df['label'].map(dz_dict)
  preds = test_df['label_desc']
  plt.figure(figsize=(8, 5))
  sns.countplot(x = test_df['label_desc'].sort_values())  

  print('Distribution:')
  return preds    