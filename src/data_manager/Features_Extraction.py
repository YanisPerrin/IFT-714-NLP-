import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import string

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# uncomment below line of code to download VADER dictionary and averaged_perceptron_tagger
# nltk.download('averaged_perceptron_tagger')
# nltk.download('vader_lexicon')
# nltk.download('universal_tagset')
# nltk.download('stopwords')
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import spacy
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

class Features_extraction():
    def __init__(self):
        self.initial_dictionary = {"ADJ":0, "ADV":0, "ADP":0, "AUX":0, "CCONJ":0, "DET":0, "INTJ":0, "NOUN":0, "NUM":0, "PART":0, "PRON":0, "PROPN":0, "PUNCT":0, "SCONJ":0, "SYM":0, "VERB":0, "X":0, "PRT":0, "CONJ":0, ".":0}
        self.text_feature = 'statements'
        
    def get_punctuation(self, text):    
        numberOfFullStops=0
        numberOfQuestionMarks=0
        numberOfExclamationMarks=0
        for line in text:
            numberOfFullStops += line.count(".")
            numberOfQuestionMarks += line.count("?")
            numberOfExclamationMarks += line.count("!")
        numberOfPunctuation = numberOfFullStops + numberOfQuestionMarks + numberOfExclamationMarks
        return numberOfPunctuation

    def get_sentiment(self, text):
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(text)
        score_negative = scores['neg'] 
        score_neutral = scores['neu'] 
        score_positive = scores['pos'] 
        return score_negative, score_neutral, score_positive

    def get_word_count(self, text):
        return len(text.split())

    def get_digits_count(self, text):
        return sum(c.isdigit() for c in text)

    def clean_text(self, text):
        text = text.lower()
        text = re.sub('.*?¿', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\w*\d\w*', '', text)
        text = re.sub('\n', '', text)
        text = re.sub('[0-9]+', '', text)
        text = re.sub(r" +", " ", text)
        return text

    def get_entities(self, text):
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(text)
        entities = [ent.text for ent in doc.ents]
        return entities

    def get_part_of_speech_count(self, text):
        dictionary = self.initial_dictionary.copy()
        tokens = word_tokenize(text.lower())
        tags = pos_tag(tokens, tagset='universal') 
        for elt in tags:
            dictionary[elt[1]]+=1 
        return dictionary
    
    def process_extraction(self, dataset):
        data = dataset.copy()
        
        # Nombre de caractere de ponctuation
        number_punctuation = data[self.text_feature].apply(self.get_punctuation)
        data['number_punctuation'] = number_punctuation
        
        # Analyse sentimentale du texte
        negative = []
        neutral = []
        positive = []
        
        for elt in tqdm(data[self.text_feature], desc = "Traiement des citations pour l'analyse sentimentale ("+dataset.name+")"):
            analysis = self.get_sentiment(elt)
            negative.append(analysis[0])
            neutral.append(analysis[1])
            positive.append(analysis[2])
            
        negative = pd.Series(negative)
        neutral = pd.Series(neutral)
        positive = pd.Series(positive)
        data['negative_value'] = negative
        data['neutral_value'] = neutral
        data['positive_value'] = positive
        
        # Nombre de mots dans le texte
        number_word = data[self.text_feature].apply(self.get_word_count)
        data['number_word'] = number_word
        
        # Nombre de chiffres dans le texte
        number_digit = data[self.text_feature].apply(self.get_digits_count)
        data['number_digit'] = number_digit
        
        # on nettoie le texte des caractères non utiles pour l'entrainement pour ne garder l'essentiel
        data[self.text_feature] = data[self.text_feature].apply(self.clean_text)
        
        # On récupère les marquage d'une partie du discours (part-of-speech : pos)
        pos_dict = data[self.text_feature].apply(lambda x: self.get_part_of_speech_count(x))
        vectorizer = DictVectorizer(sort=False, sparse=False)
        pos_values = vectorizer.fit_transform(pos_dict)
        for i, column in enumerate(self.initial_dictionary.keys()):
            data[column] = pos_values[:,i]
        data.drop(".", axis=1, inplace=True)
        
        return data
    
    # Fonction qui fit le TfidfVectorizer() à partir du vocabulaire des données entrées en paramètres
    def fit_tfif(self, data, ngram_range):
        self.tfidf_vectorizer = TfidfVectorizer(ngram_range=ngram_range).fit(data[self.text_feature])
        
    # Fonction qui récupére la représentation TF-IDF
    def get_tfif_representation(self, data):
        new_statements = self.tfidf_vectorizer.transform(data[self.text_feature])
        return new_statements.toarray()
    
    def get_sentence_embedding(self, data):
        model = SentenceTransformer('all-mpnet-base-v2')
        sentences = data[self.text_feature].values
        sentence_embeddings = model.encode(sentences)
        return sentence_embeddings
    
    def get_doc2vec_representation(self, data):
        data_sentences = data[self.text_feature].tolist()
        data_tokenized_sentences = [word_tokenize(sentence) for sentence in data_sentences]
        tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(data_tokenized_sentences)]
        model = Doc2Vec(vector_size=100, window=5, min_count=1, workers=4, epochs=10)
        model.build_vocab(tagged_data)
        model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
        doc2vec_representation = [model.infer_vector(doc.words) for doc in tagged_data]
        return doc2vec_representation
