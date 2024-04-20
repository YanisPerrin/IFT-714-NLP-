from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

class Evaluator():

    def __init__(self):
        self.str_representations = ["SE", "TF-IDF", "Doc2Vec"]

    def get_metrics(self, models, str_models, all_pairs_Xtrain_y_train, all_pairs_Xtest_y_test):
        
        # dictionnaire pour stocker les métriques
        all_accuracy_bert = {}
        all_f1_score_bert = {}
        all_recall_score_bert = {}
        all_precision_score_bert = {}

        for j in tqdm(range(len(all_pairs_Xtest_y_test)), desc="Traitement pour les différentes représentation du texte") :
            X_train = all_pairs_Xtrain_y_train[j][0]
            y_train = all_pairs_Xtrain_y_train[j][1]
            X_test = all_pairs_Xtest_y_test[j][0]
            y_test = all_pairs_Xtest_y_test[j][1] 
            representation = self.str_representations[j]
            for i, model in enumerate(models):
                name_model = str_models[i]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy_model = round(accuracy_score(y_test, y_pred),2)
                f1_score_model = round(f1_score(y_test, y_pred, average = "weighted"),2)
                recall_score_model = round(recall_score(y_test, y_pred, average = "weighted"),2)
                precision_score_model = round(precision_score(y_test, y_pred, average = "weighted"),2)
                print(f"{name_model} - ({representation}): {accuracy_model*100}%")
                all_accuracy_bert[f"{name_model} - ({representation})"] =  accuracy_model
                all_f1_score_bert[f"{name_model} - ({representation})"] =  f1_score_model
                all_recall_score_bert[f"{name_model} - ({representation})"] =  recall_score_model
                all_precision_score_bert[f"{name_model} - ({representation})"] =  precision_score_model
        return all_accuracy_bert, all_f1_score_bert, all_recall_score_bert, all_precision_score_bert