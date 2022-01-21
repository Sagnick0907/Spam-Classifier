# Importing the Dataset

import pandas as pd

messages = pd.read_csv('C:/Users/91943/PROJECT/Spam Classifier/smsspamcollection/SMSSpamCollection',
                       sep='\t', names=["label", "message"])

#Data cleaning and preprocessing
import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i]) # remove unnecessary symbols & numbers
    review = review.lower()                                   # lower case all sentences
    review = review.split()                                   # split sentences into words

    review = [ps.stem(word) for word in review if not word in stopwords.words('english')] #stemming & removal of stopwords
    review = ' '.join(review)
    corpus.append(review)
    
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2000)                      #  top 5000 frequent features are selected
X = cv.fit_transform(corpus).toarray()

y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values


# Train | Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training model using Naive Bayes classifier

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)
y_pred=spam_detect_model.predict(X_test)

#Performance Check

from sklearn.metrics import confusion_matrix, precision_score
confusion_m = confusion_matrix(y_test,y_pred)
precision_m = precision_score(y_test, y_pred, average='binary')

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)


















