import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
import json
import random
from keras.models import load_model
ERROR_THRESHOLD = 0.5

model = load_model('chatbot_Application_model.h5')

intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
labels = pickle.load(open('labels.pkl','rb'))


def bank_of_words(s,words, show_details=True):
    bag_of_words = [0 for _ in range(len(words))]
    sent_words = nltk.word_tokenize(s)
    sent_words = [lemmatizer.lemmatize(word.lower()) for word in sent_words]
    for sent in sent_words:
        for i,w in enumerate(words):
            if w == sent:
                bag_of_words[i] = 1
    return np.array(bag_of_words)

def predict_label(s, model):
    # filtering out predictions
    pred = bank_of_words(s, words,show_details=False)
    response = model.predict(np.array([pred]))[0]
    ERROR_THRESHOLD = 0.25
    final_results = [[i,r] for i,r in enumerate(response) if r>ERROR_THRESHOLD]
    final_results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in final_results:
        return_list.append({"intent": labels[r[0]], "probability": str(r[1])})
    return return_list

def Response(ints, intents_json):
    tags = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tags):
            response = random.choice(i['responses'])
            break
    return response


"""
def chatbot_response(msg):
    ints = predict_label(msg, model)
    response = Response(ints, intents)
    return response

"""

def chatbot_response(msg):
    # Get list of predicted intents and their probabilities
    ints = predict_label(msg, model)
    # Iterate through predicted intents to find the one with highest probability
    max_prob_intent = None
    max_prob = 0
    for i in ints:
        if float(i['probability']) > max_prob:
            max_prob_intent = i
            max_prob = float(i['probability'])
    # If no intent has probability above the error threshold, return default response
    if max_prob < ERROR_THRESHOLD:
        return "I'm sorry, I didn't understand that. Can you please try again?"
    # Otherwise, return a response from the intent with the highest probability
    else:
        response = Response([max_prob_intent], intents)
        return response


