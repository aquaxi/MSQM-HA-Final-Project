from IPython.display import display, Markdown, HTML
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.metrics import log_loss, accuracy_score, classification_report, f1_score, roc_auc_score
import scikitplot as skplt
import seaborn as sns
import matplotlib.pyplot as plt

#!/usr/bin/env python
# coding: utf-8

# In[ ]:

def direct_match(keyword, document):
  """ Count of substring keyword matches in a document

  This function counts the number of keyword matches in a document.
  This is a substring match and is case-sensitive.
  Example if keyword = 'tax' and document = 'tax on taxes' then
  the number of matches would be = 2.

  Args:
    keyword  (str): String to match on
    document (str): Document searched for matches

  Returns:
    int: Count of keyword matches in document
  """
  return document.count(keyword)


def get_top_n(bm25, query, n=5):
    """ Get the top N ranking documents that match the query
        
    Args: 
     bm25 : bm25 model
     query (str): query text
     n (int): number of document that would like to display
    
    Returns:
     list of index correlated with the top ranking document that match query
    """    
    scores = np.array(bm25.get_scores(query))    
    idx = np.argpartition(scores, -n)[-n:]  

    return idx[np.argsort(-scores[idx])]

def mark(s, color='black'):
      return "<text style=color:{}>{}</text>".format(color, s)

def highlight(keywords, tokens, color='SteelBlue'):

    kw_set = set(keywords)
    tokens_hl = []    
    for t in tokens:
        if t in kw_set:
            tokens_hl.append('<b>'+mark(t, color=color)+'</b>')
        else:
            tokens_hl.append(t)
    
    return " ".join(tokens_hl)

def color_label(labels):
  """ Color the label according to the classes  
    
    Args:
       Labels (list): list of classes (0-4) 
    
    return:
       display with html format 
  """  
  color = {
        0: 'Olive', #Cancer
        1: 'Gold', #GI
        2: 'SlateBlue', #CNS
        3: 'DeepPink', #CV
        4: 'SlateGray' #general
    }
  label_token = []
  for i in labels:    
    label_token.append(mark(dz_dict[i], color[i]))  
  return display(HTML('<h3>Label:' + ', '.join(label_token)))

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
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
   
    eval_model(X, y, model, probas = True)
    

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