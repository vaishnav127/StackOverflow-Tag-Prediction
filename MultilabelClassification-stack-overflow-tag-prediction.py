
# coding: utf-8

# # Predict tags on StackOverflow with linear models

# ### Libraries
# 
# In this you will need the following libraries:
# - [Numpy](http://www.numpy.org) — a package for scientific computing.
# - [Pandas](https://pandas.pydata.org) — a library providing high-performance, easy-to-use data structures and data analysis tools for the Python
# - [scikit-learn](http://scikit-learn.org/stable/index.html) — a tool for data mining and data analysis.
# - [NLTK](http://www.nltk.org) — a platform to work with natural language.

# ### Data
# 
# The following cell will download all data required for this assignment into the folder `week1/data`.

# In[2]:


import sys
sys.path.append("..")
from common.download_utils import download_week1_resources

download_week1_resources()




# ### Text preprocessing

# For this and most of the following assignments you will need to use a list of stop words. It can be downloaded from *nltk*:

# In[3]:


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
print("done")


# In this you will deal with a dataset of post titles from StackOverflow. You are provided a split to 3 sets: *train*, *validation* and *test*. All corpora (except for *test*) contain titles of the posts and corresponding tags (100 tags are available). The *test* set is provided for Coursera's grading and doesn't contain answers. Upload the corpora using *pandas* and look at the data:

# In[4]:


from ast import literal_eval
import pandas as pd
import numpy as np
print("done")


# In[5]:


def read_data(filename):
    data = pd.read_csv(filename, sep='\t')
    data['tags'] = data['tags'].apply(literal_eval)
    return data

print("done")


# In[6]:


train = read_data('data/train.tsv')
validation = read_data('data/validation.tsv')
test = pd.read_csv('data/test.tsv', sep='\t')
print("done")


# In[10]:


train.head()


# In[11]:


validation.iloc[4]['title']


# As you can see, *title* column contains titles of the posts and *tags* colum countains the tags. It could be noticed that a number of tags for a post is not fixed and could be as many as necessary.

# For a more comfortable usage, initialize *X_train*, *X_val*, *X_test*, *y_train*, *y_val*.

# In[7]:


X_train, y_train = train['title'].values, train['tags'].values
X_val, y_val = validation['title'].values, validation['tags'].values
X_test = test['title'].values
print("done")


# One of the most known difficulties when working with natural data is that it's unstructured. For example, if you use it "as is" and extract tokens just by splitting the titles by whitespaces, you will see that there are many "weird" tokens like *3.5?*, *"Flip*, etc. To prevent the problems, it's usually useful to prepare the data somehow. In this  you'll write a function, which will be also used in the other assignments. 
# 
# ** (TextPrepare).** Implement the function *text_prepare* following the instructions. After that, run the function *test_test_prepare* to test it on tiny cases and submit it to Coursera.

# In[8]:


import re
from nltk.tokenize import word_tokenize
nltk.download('punkt')

print("done")


# In[9]:


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]') 
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]') # take all words that contain characters other than 0-9,a-z,#,+
STOPWORDS = set(stopwords.words('english'))

def text_prepare(text):
    """
        text: a string
        
        return: modified initial string
    """
    #text = # lowercase text
    text =text.lower()
    #text = # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = re.sub(REPLACE_BY_SPACE_RE, ' ', text)
    #text = # delete symbols which are in BAD_SYMBOLS_RE from text
    text =  re.sub(BAD_SYMBOLS_RE, '', text)
    #text = # delete stopwords from text
    token_word=word_tokenize(text)
    filtered_sentence = [w for w in token_word if not w in STOPWORDS] # filtered_sentence contain all words that are not in stopwords dictionary
    lenght_of_string=len(filtered_sentence)
    text_new=""
    for w in filtered_sentence:
        if w!=filtered_sentence[lenght_of_string-1]:
             text_new=text_new+w+" " # when w is not the last word so separate by whitespace
        else:
            text_new=text_new+w
            
    text = text_new
    return text
print("done")


# In[10]:


def test_text_prepare():
    examples = ["SQL Server - any equivalent of Excel's CHOOSE function?",
                "How to free c++ memory vector<int> * arr?"]
    answers = ["sql server equivalent excels choose function", 
               "free c++ memory vectorint arr"]
    for ex, ans in zip(examples, answers):
        if text_prepare(ex) != ans:
            return "Wrong answer for the case: '%s'" % ex
    return 'Basic tests are passed.'

print("done")


# In[11]:


print(test_text_prepare())


# Run your implementation for questions from file *text_prepare_tests.tsv* to earn the points.

# In[12]:


prepared_questions = []
for line in open('data/text_prepare_tests.tsv', encoding='utf-8'):
    line = text_prepare(line.strip())
    prepared_questions.append(line)
text_prepare_results = '\n'.join(prepared_questions)

grader.submit_tag('TextPrepare', text_prepare_results)
print("done")


# Now we can preprocess the titles using function *text_prepare* and  making sure that the headers don't have bad symbols:

# In[13]:


X_train = [text_prepare(x) for x in X_train]
X_val = [text_prepare(x) for x in X_val]
X_test = [text_prepare(x) for x in X_test]

print("done")


# In[14]:


print(len(X_train))

import collections 
from collections import Counter
import re

words=[]
tag_w=[]
for i in range(0,100000):
    #print(i)
    words = words+(re.findall(r'\w+', X_train[i])) # words cantain all the words in the dataset
    tag_w=tag_w+y_train[i] # tage_w contain all tags that aree present in train dataset

print("done") 
words_counts = Counter(words) # counter create the dictinary of unique words with their frequncy
tag_counts=Counter(tag_w)
#print(words_counts)
#print(tag_counts)


# For each tag and for each word calculate how many times they occur in the train corpus. 
# 
# **2 (WordsTagsCount).** Find 3 most popular tags and 3 most popular words in the train data and submit the results to earn the points.

# In[16]:


# Dictionary of all tags from train corpus with their counts.
tags_counts = tag_counts
# Dictionary of all words from train corpus with their counts.
words_counts = Counter(words)
#print(tags_counts)
#print(tags_counts.keys())


# We are assume that *tags_counts* and *words_counts* are dictionaries like `{'some_word_or_tag': frequency}`. After appllying the sorting procedure, results will be look like this: `[('most_popular_word_or_tag', frequency), ('less_popular_word_or_tag', frequency), ...]`. The grader gets the results in the following format (two comma-separated strings with line break):
# 
#     tag1,tag2,tag3
#     word1,word2,word3
# 
# Pay attention that in this assignment you should not submit frequencies or some additional information.

# In[17]:


most_common_tags = sorted(tags_counts.items(), key=lambda x: x[1], reverse=True)[:3]
most_common_words = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:3]


print(most_common_words)
grader.submit_tag('WordsTagsCount', '%s\n%s' % (','.join(tag for tag, _ in most_common_tags), 
                                                ','.join(word for word, _ in most_common_words)))

print("done")


# ### Transforming text to a vector
# 
# Machine Learning algorithms work with numeric data and we cannot use the provided text data "as is". There are many ways to transform text data to numeric vectors. In this you will try to use two of them.
# 
# #### Bag of words
# 
# One of the well-known approaches is a *bag-of-words* representation. To create this transformation, follow the steps:
# 1. Find *N* most popular words in train corpus and numerate them. Now we have a dictionary of the most popular words.
# 2. For each title in the corpora create a zero vector with the dimension equals to *N*.
# 3. For each text in the corpora iterate over words which are in the dictionary and increase by 1 the corresponding coordinate.
# 
# Let's try to do it for a toy example. Imagine that we have *N* = 4 and the list of the most popular words is 
# 
#     ['hi', 'you','me', 'are']
# 
# Then we need to numerate them, for example, like this: 
# 
#     {'hi': 0, 'you': 1, 'me': 2, 'are': 3}
# 
# And we have the text, which we want to transform to the vector:
# 
#     'hi how are you'
# 
# For this text we create a corresponding zero vector 
# 
#     [0, 0, 0, 0]
#     
# And interate over all words, and if the word is in the dictionary, we increase the value of the corresponding position in the vector:
# 
#     'hi':  [1, 0, 0, 0]
#     'how': [1, 0, 0, 0] # word 'how' is not in our dictionary
#     'are': [1, 0, 0, 1]
#     'you': [1, 1, 0, 1]
# 
# The resulting vector will be 
# 
#     [1, 1, 0, 1]
#    
# Implement the described encoding in the function *my_bag_of_words* with the size of the dictionary equals to 5000. To find the most common words use train data. You can test your code using the function *test_my_bag_of_words*.

# In[18]:


DICT_SIZE = 5000
most_common_words = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:5000] #most_common_words contain 5000 words in sorted order of frequncy
WORDS_TO_INDEX={}
INDEX_TO_WORDS={}
for i in range(0,5000):
    WORDS_TO_INDEX[most_common_words[i][0]]=i   # most_common_words[i][0] means extracting ith word from the dictioaanry, words to index conatain the index value of the word
    INDEX_TO_WORDS[i]=most_common_words[i][0] # index to word conatain the word conrrespond to the index



    
#INDEX_TO_WORDS = ####### YOUR CODE HERE #######
ALL_WORDS = WORDS_TO_INDEX.keys()

def my_bag_of_words(text, words_to_index, dict_size):
    """
        text: a string
        dict_size: size of the dictionary
        
        return a vector which is a bag-of-words representation of 'text'
    """
    result_vector = np.zeros(dict_size)
    y=text.split(" ")
    for i in range(0,len(y)):
        for key,value in words_to_index.items():
            if y[i]==key:
                result_vector[words_to_index[key]]=result_vector[words_to_index[key]]+1  #  result_vector[words_to_index[key]] conatin the count of the presence of  word in the text
            
    return result_vector # result vector is the vector of the size of the no of words taken as features having count of then in the text

print("dsf")


# In[19]:


def test_my_bag_of_words():
    words_to_index = {'hi': 0, 'you': 1, 'me': 2, 'are': 3}
    examples = ['hi how are you']
    answers = [[1, 1, 0, 1]]
    for ex, ans in zip(examples, answers):
        if (my_bag_of_words(ex, words_to_index, 4) != ans).any():
            return "Wrong answer for the case: '%s'" % ex
    return 'Basic tests are passed.'


# In[20]:


print(test_my_bag_of_words())


# Now apply the implemented function to all samples (this might take up to a minute):

# In[21]:


from scipy import sparse as sp_sparse


# In[22]:


X_train_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_train])
X_val_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_val])
X_test_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_test])
print('X_train shape ', X_train_mybag.shape)
print('X_val shape ', X_val_mybag.shape)
print('X_test shape ', X_test_mybag.shape)


# As you might notice, we transform the data to sparse representation, to store the useful information efficiently. There are many [types](https://docs.scipy.org/doc/scipy/reference/sparse.html) of such representations, however slkearn algorithms can work only with [csr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix) matrix, so we will use this one.

# **(BagOfWords).** For the 10th row in *X_train_mybag* find how many non-zero elements it has. In this  the answer (variable *non_zero_elements_count*) should be a number, e.g. 20.

# In[23]:


row = X_train_mybag[10].toarray()[0]
non_zero_elements_count=0
for i in range(0,5000):
    if (row[i]==1):
        non_zero_elements_count=non_zero_elements_count+1
    

print(non_zero_elements_count)
#non_zero_elements_count = ####### YOUR CODE HERE #######

grader.submit_tag('BagOfWords', str(non_zero_elements_count))


# #### TF-IDF
# 
# The second approach extends the bag-of-words framework by taking into account total frequencies of words in the corpora. It helps to penalize too frequent words and provide better features space. 
# 
# Implement function *tfidf_features* using class [TfidfVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) from *scikit-learn*. Use *train* corpus to train a vectorizer. Don't forget to take a look into the arguments that you can pass to it. We suggest that you filter out too rare words (occur less than in 5 titles) and too frequent words (occur more than in 90% of the titles). Also, use bigrams along with unigrams in your vocabulary. 

# In[24]:


from sklearn.feature_extraction.text import TfidfVectorizer
print(X_train[:3])


# In[25]:


def tfidf_features(X_train, X_val, X_test):
    """
        X_train, X_val, X_test — samples        
        return TF-IDF vectorized representation of each sample and vocabulary
    """
    # Create TF-IDF vectorizer with a proper parameters choice
    # Fit the vectorizer on the train set
    # Transform the train, test, and val sets and return the result
    
    tfidf_vectorizer =  TfidfVectorizer(min_df=5,max_df=0.9,ngram_range=(1,2),token_pattern= '(\S+)')#  '(\S+)'  means any no white space
    X_train=tfidf_vectorizer.fit_transform(X_train)
    X_val=tfidf_vectorizer.transform(X_val)
    X_test=tfidf_vectorizer.transform(X_test)
    ######################################
    ######### YOUR CODE HERE #############
    ######################################
    
    return X_train, X_val, X_test, tfidf_vectorizer.vocabulary_

#tfidf_vectorizer.vocabulary_ returns just index of feature


# Once you have done text preprocessing, always have a look at the results. Be very careful at this step, because the performance of future models will drastically depend on it. 
# 
# In this case, check whether you have c++ or c# in your vocabulary, as they are obviously important tokens in our tags prediction :

# In[26]:


X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vocab = tfidf_features(X_train, X_val, X_test)
tfidf_reversed_vocab = {i:word for word,i in tfidf_vocab.items()}


# In[27]:


######### YOUR CODE HERE #############
#print(X_train_tfidf[2])
print('X_test_tfidf ', X_test_tfidf.shape) 
print('X_val_tfidf ',X_val_tfidf.shape)


# If you can't find it, we need to understand how did it happen that we lost them? It happened during the built-in tokenization of TfidfVectorizer. Luckily, we can influence on this process. Get back to the function above and use '(\S+)' regexp as a *token_pattern* in the constructor of the vectorizer.  

# Now, use this transormation for the data and check again.

# In[89]:


print(tfidf_vocab)


# ### MultiLabel classifier
# 
# As we have noticed before, in this each example can have multiple tags. To deal with such kind of prediction, we need to transform labels in a binary form and the prediction will be a mask of 0s and 1s. For this purpose it is convenient to use [MultiLabelBinarizer](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html) from *sklearn*.

# In[28]:


from sklearn.preprocessing import MultiLabelBinarizer


# In[29]:


mlb = MultiLabelBinarizer(classes=sorted(tags_counts.keys()))
y_train = mlb.fit_transform(y_train) # it chnage the y_train in feature form like alll clases with 0,1 value
y_val = mlb.fit_transform(y_val)


# Implement the function *train_classifier* for training a classifier. In this  we suggest to use One-vs-Rest approach, which is implemented in [OneVsRestClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html) class. In this approach *k* classifiers (= number of tags) are trained. As a basic classifier, use [LogisticRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html). It is one of the simplest methods, but often it performs good enough in text classification s. It might take some time, because a number of classifiers to train is large.

# In[30]:


from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier


# In[31]:


def train_classifier(X_train, y_train):
    """
      X_train, y_train — training data
      
      return: trained classifier
    """
    
    # Create and fit LogisticRegression wraped into OneVsRestClassifier.
    model=OneVsRestClassifier(LogisticRegression()).fit(X_train,y_train)
    
    return model

    ######################################
    ######### YOUR CODE HERE #############
    ######################################   
    
print('X_test_tfidf ', X_test_tfidf.shape) 
print('X_val_tfidf ',X_val_tfidf.shape)


# Train the classifiers for different data transformations: *bag-of-words* and *tf-idf*.

# In[32]:


classifier_mybag = train_classifier(X_train_mybag, y_train)
classifier_tfidf = train_classifier(X_train_tfidf, y_train)


# Now you can create predictions for the data. You will need two types of predictions: labels and scores.

# In[33]:


y_val_predicted_labels_mybag = classifier_mybag.predict(X_val_mybag) #y_val_predicted_labels_mybag is in the same format of y_train
y_val_predicted_scores_mybag = classifier_mybag.decision_function(X_val_mybag)

y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)


# Now take a look at how classifier, which uses TF-IDF, works for a few examples:


y_val_pred_inversed = mlb.inverse_transform(y_val_predicted_labels_tfidf) # just opposite of tranform means it will give the name of classes rather than 0,1 in classes
y_val_inversed = mlb.inverse_transform(y_val)
for i in range(10):
    print('Title:\t{}\nTrue labels:\t{}\nPredicted labels:\t{}\n\n'.format(
        X_val[i],
        ','.join(y_val_inversed[i]),
        ','.join(y_val_pred_inversed[i])
    ))


# Now, we would need to compare the results of different predictions, e.g. to see whether TF-IDF transformation helps or to try different regularization techniques in logistic regression. For all these experiments, we need to setup evaluation procedure. 

# ### Evaluation
# 
# To evaluate the results we will use several classification metrics:
#  - [Accuracy](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)
#  - [F1-score](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
#  - [Area under ROC-curve](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
#  - [Area under precision-recall curve](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score) 
#  
# Make sure you are familiar with all of them. How would you expect the things work for the multi-label scenario? Read about micro/macro/weighted averaging following the sklearn links provided above.

# In[35]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score


# Implement the function *print_evaluation_scores* which calculates and prints to stdout:
#  - *accuracy*
#  - *F1-score macro/micro/weighted*
#  - *Precision macro/micro/weighted*

# In[36]:


def print_evaluation_scores(y_val, predicted):
    accuracy=accuracy_score(y_val, predicted)
    f1_score_macro=f1_score(y_val, predicted, average='macro')
    f1_score_micro=f1_score(y_val, predicted, average='micro')
    f1_score_weighted=f1_score(y_val, predicted, average='weighted')
    precision_macro=average_precision_score(y_val, predicted, average='macro')
    precision_micro=average_precision_score(y_val, predicted, average='micro')
    precision_weighted=average_precision_score(y_val, predicted, average='weighted')
    print(accuracy,f1_score_macro,f1_score_micro,f1_score_weighted,precision_macro,precision_micro,precision_weighted)
    
    


# In[37]:


print('Bag-of-words')
print_evaluation_scores(y_val, y_val_predicted_labels_mybag)
print('Tfidf')
print_evaluation_scores(y_val, y_val_predicted_labels_tfidf)


# You might also want to plot some generalization of the [ROC curve](http://scikit-learn.org/stable/modules/model_evaluation.html#receiver-operating-characteristic-roc) for the case of multi-label classification. Provided function *roc_auc* can make it for you. The input parameters of this function are:
#  - true labels
#  - decision functions scores
#  - number of classes

# In[38]:


from metrics import roc_auc
get_ipython().run_line_magic('matplotlib', 'inline')


# In[39]:


n_classes = len(tags_counts)
roc_auc(y_val, y_val_predicted_scores_mybag, n_classes)


# In[40]:


n_classes = len(tags_counts)
roc_auc(y_val, y_val_predicted_scores_tfidf, n_classes)


# - compare the quality of the bag-of-words and TF-IDF approaches and chose one of them.
# - for the chosen one, try *L1* and *L2*-regularization techniques in Logistic Regression with different coefficients (e.g. C equal to 0.1, 1, 10, 100).
# 
# You also could try other improvements of the preprocessing / model, if you want. 

# In[53]:


from sklearn.pipeline import make_pipeline
print("done") 
def train_classifier(X_train, y_train):
    """
      X_train, y_train — training data
      
      return: trained classifier
    """
    
    # Create and fit LogisticRegression wraped into OneVsRestClassifier.
   # pipe = make_pipeline( LogisticRegression(penalty="l1"),OneVsRestClassifier(LogisticRegression(penalty="l2")))
    #model=pipe.fit(X_train, y_train)     
    model=OneVsRestClassifier(LogisticRegression(penalty="l1",C=1)).fit(X_train,y_train)
    
    return model

print("done1")    

classifier_mybag = train_classifier(X_train_mybag, y_train)
#classifier_tfidf = train_classifier(X_train_tfidf, y_train)
print("done2")
y_val_predicted_labels_mybag = classifier_mybag.predict(X_val_mybag)
#y_val_predicted_scores_mybag = classifier_mybag.decision_function(X_val_mybag)
print("done3")
#y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
#y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score

def print_evaluation_scores(y_val, predicted):
    accuracy=accuracy_score(y_val, predicted)
    f1_score_macro=f1_score(y_val, predicted, average='macro')
    f1_score_micro=f1_score(y_val, predicted, average='micro')
    f1_score_weighted=f1_score(y_val, predicted, average='weighted')
    precision_macro=average_precision_score(y_val, predicted, average='macro')
    precision_micro=average_precision_score(y_val, predicted, average='micro')
    precision_weighted=average_precision_score(y_val, predicted, average='weighted')
    print(accuracy,f1_score_macro,f1_score_micro,f1_score_weighted,precision_macro,precision_micro,precision_weighted)

print("done4")    
print('Bag-of-words')
print_evaluation_scores(y_val, y_val_predicted_labels_mybag)
#print('Tfidf')
#print_evaluation_scores(y_val, y_val_predicted_labels_tfidf)    


# When you are happy with the quality, create predictions for *test* set, which you will submit to Coursera.




test_predictions = classifier_mybag.predict(X_test_mybag)
test_pred_inversed = mlb.inverse_transform(test_predictions)

test_predictions_for_submission = '\n'.join('%i\t%s' % (i, ','.join(row)) for i, row in enumerate(test_pred_inversed))
grader.submit_tag('MultilabelClassification', test_predictions_for_submission)








