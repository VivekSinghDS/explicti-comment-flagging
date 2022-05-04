import pickle
import time
import re
import string
import nltk
import os
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

## fastapi conversion
import uvicorn
from fastapi import FastAPI, status
from pydantic import BaseModel


with open('./classifier_model.pickle', 'rb') as file:
    classifier = pickle.load(file)
    file.close()


with open('./word_vectorizer.pickle', 'rb') as file:
    word_vectorizer = pickle.load(file)
    file.close()


def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"\r", "", text)
    text = re.sub(r"he's", "he is", text)

    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)

    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub("(\\W)", " ", text)
    text = re.sub('\S*\d\S*\s*', '', text)

    return text


nltk.download('stopwords')
sn = SnowballStemmer(language='english')


def stem_stopwords_reduction(text):
    text = [sn.stem(w) for w in text.split() if not w in set(stopwords.words('english'))]
    return ' '.join(text)


def make_test_predictions(text, classifier):
    df = {'id': [123], 'comment_text': [text]}
    df = pd.DataFrame(df)
    df.comment_text = df.comment_text.apply(clean_text)
    df.comment_text = df.comment_text.apply(stem_stopwords_reduction)
    X_test = df.comment_text
    X_test_transformed = word_vectorizer.transform(X_test)
    y_test_pred = classifier.predict_proba(X_test_transformed)

    score_dict = {0: 'Toxic', 1: 'Severely Toxic', 2: 'Obscene', 3: 'Threat', 4: 'Insult', 5: 'Identity Hate'}
    offensive_score = np.amax(y_test_pred)

    if (offensive_score > 0.7):

        print('Your comment is classified as {}, we suggest you to use a relatively normal tone'.format(
            score_dict[np.argmax(y_test_pred)]))

        return score_dict[np.argmax(y_test_pred)]

    elif (offensive_score > 0.5):
        print("Your comment might be taken as offensive to some of our users, kindly change it")
        return score_dict[np.argmax(y_test_pred)]

    else:
        print('The comment/statement has no issues ')
        return 'No issues with it '





app = FastAPI()
class Item(BaseModel):
    input_path:str



# INPUT_PATH = 'smart_cropping_grid/test/img6.jpg'
# if(INPUT_PATH.split('.')[-1]=='jpg'):
#     im = Image.open(INPUT_PATH)
#
#     new_path = INPUT_PATH[:-1]+'eg'
#
#     im.save(new_path)
#     INPUT_PATH = new_path



# smart_crop(INPUT_PATH, OUTPUT_PATH)
# os.remove(INPUT_PATH)



@app.post('/comment_identifier/')
async def create_item(item:Item):
    item_dict = item.dict()

    # test('test/img3.jpeg')
    x = make_test_predictions(item_dict['input_path'], classifier)
    return x



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7000)