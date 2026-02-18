from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

data=pd.read_csv(r"C:\Users\aslam\OneDrive\Desktop\spam\dataset.csv")


#text and labels
texts=data["text"]
labels=data["text_type"]


#train test split
x_train,x_test,y_train,y_test=train_test_split(texts,labels,test_size=0.25,random_state=42)

#pipeline

pipe= Pipeline([
    ('vectorization',CountVectorizer()),
    ('classifier',MultinomialNB())
])

#train the model
pipe.fit(x_train,y_train)

#predict
preds=pipe.predict(x_test)

#evaluate model
print("test predictions",preds)
print("actual labels",y_test)
print("accuracy:",accuracy_score(y_test,preds))

#try a new example
new_text=["Congratulations! 🎉 You won ₹50,000. Click here to claim now: http://bit.ly/free-money"]
new_pred=pipe.predict(new_text)
print("new text:",new_text[0])
print("predicted label:",new_pred[0])


import pickle

pickle.dump(pipe, open("spam_model.pkl", "wb"))