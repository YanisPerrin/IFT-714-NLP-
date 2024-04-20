import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import sys
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir) 

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# uncomment below line of code to download VADER dictionary and averaged_perceptron_tagger
# nltk.download('averaged_perceptron_tagger')
# nltk.download('vader_lexicon')
# nltk.download('universal_tagset')
# nltk.download('stopwords')
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# pip install spacy
#!python -m spacy download en
from tqdm import tqdm

class Preprocessor():
    def __init__(self):
        
        # Création d'un dictionnaire pour soustraire l'étiquette actuelle de l'historique de crédit
        self.credit_dictionary = {
            2 : "c_half_true",
            3 : "c_mostly_true",
            1 : "c_false",
            0 : "c_barely_true",
            4 : "c_pants_on_fire"
        }
        # On crée la liste des vrais noms d'attributs (la liste initiale ne contient pas les bons)
        self.true_columns = [
            'id_citation',           
            'labels',             
            'statements',        
            'subjects',           
            'speaker',             
            'function',           
            'state',               
            'affiliations',        
            'c_barely_true',      
            'c_false',            
            'c_half_true',        
            'c_mostly_true',      
            'c_pants_on_fire',    
            'context'
        ]
        
        self.binary_map = {
            5: 1,
            3: 1, 
            2: 1,
            0: 1,
            1: 0,
            4: 0 
        }
        self.target = 'labels'

        self.max_sequence_length = 0
        
    # Fonction qui renomme correctement les attributs  
    def rename_columns(self, all_datas):
        all_dict = []
        for data in all_datas:
            all_dict.append(dict(zip(data, self.true_columns)))
        for i, data in enumerate(all_datas):
            data.rename(columns = all_dict[i], inplace=True)
    
    # Fonction qui applique le prétraitement du texte (tokenisation + lemmatisation)
    def preprocessing_text(self, text):
        # On récupère les jetons
        tokens = word_tokenize(text.lower())

        # On sélectionne uniquement les jetons qui ne sont pas dans la banque de mots vides
        filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]

        # instanciation of a WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()

        # apply the lemmatization
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

        # add space
        processed_text = ' '.join(lemmatized_tokens)

        return processed_text

    # Fonction pour soustraire l'étiquette actuelle de l'historique de crédit
    def substraction_credit(self, data, has_credit=True):
        new_dataset = data.copy()
        if has_credit:
            for i in range(new_dataset.shape[0]):
                key = new_dataset[self.target].loc[i]
                if key == 5: continue
                col = self.credit_dictionary[key]
                new_dataset.loc[i,col] -= 1
        return new_dataset
    
    # Création d'une fonction qui assemble les différentes étapes du prétraitement
    def preprocessing_target(self, data, has_credit=True):
        new_dataset = data.copy()
        # encodage de la cible
        le = LabelEncoder()
        new_dataset[self.target] = le.fit_transform(new_dataset[self.target])

        # On soustrait l'étiquette actuelle de l'historique de crédit au compte actuelle de cette étiquette via la fonction substraction_credit
        return self.substraction_credit(new_dataset, has_credit)
    
    def tranform_into_binary_classification(self, data):
        dataset = data.copy()
        dataset[self.target] = dataset[self.target].map(self.binary_map)
        return dataset
    
    def fit_keras_tokenizer(self, data, column):
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(data[column])

    def get_padded_senquences(self, data, column):
        sequences = self.tokenizer.texts_to_sequences(data[column])
        if data.name == 'Entrainement':
            self.max_sequence_length = max(len(seq) for seq in sequences)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_sequence_length, padding='post')
        return padded_sequences
    
    def get_max_senquence_length(self):
        return self.max_sequence_length