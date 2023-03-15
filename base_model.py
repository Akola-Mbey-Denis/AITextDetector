import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,CountVector,CountVectorizer
from sklearn.linear_model import LogisticRegression
import csv
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline


# Read The data
training_set = pd.read_json('dataset/train_set_processed.json')
test_set = pd.read_json('dataset/test_set_processed.jsontest_set.json')
X = training_set['Text']
y = train_set['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)



classifier = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer(max_features=10000)),
                     ('clf', MultinomialNB())])

predictions = clf.predict(X_test)



# Write predictions to a file
with open("submission.csv", "w") as pred:
    csv_out = csv.writer(pred)
    csv_out.writerow(['id','label'])
    for i, row in enumerate(predictions):
        csv_out.writerow([i, row])
        