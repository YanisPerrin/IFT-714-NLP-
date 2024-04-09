import random
from tqdm import tqdm
import nlpaug.augmenter.word as naw
import pandas as pd

class Data_Augmentation():
    def apply_eda(self, eda, row, alpha=0.2):

        text = row['statements']
        l = len(text.split(" "))
        operation = {
            'synonym_replacement' : eda.synonym_replacement(text, n=round(alpha * l), top_n=10),
            'random_insertion' : eda.random_insertion(text, n=round(alpha * l)),
            'random_deletion' : eda.random_deletion(text, p=alpha),
            'random_swap' : eda.random_swap(text, n=round(alpha * l))
        }

        operations = [
            eda.synonym_replacement(text, top_n=10),
            eda.random_insertion(text),
            eda.random_deletion(text, p=0.2),
            eda.random_swap(text)
        ]

        augmented_text = random.choice(operations)

        augmented_row = [
            row['labels'],
            augmented_text,
            row['c_barely_true'],
            row['c_false'],
            row['c_half_true'],
            row['c_mostly_true'],
            row['c_pants_on_fire']
        ]

        return augmented_row

    def process_augment_eda(self, eda, data, N, alpha):
        augmented_data = data.copy()

        indexes = random.sample(range(len(data)), N)
        for index in tqdm(indexes, desc = "Augmentation des données"):
            row = data.loc[index]
            augmented_row = self.apply_eda(eda, row, alpha)

            augmented_data.loc[len(augmented_data)] = augmented_row

        return augmented_data
    
    def bert_augmentation(self, text, act="substitute", p=0.3):
        TOPK=20 #default=100
        aug_bert = naw.ContextualWordEmbsAug(
            model_path='distilbert-base-uncased', 
            device='cuda', aug_p=p,
            action=act, top_k=TOPK)
        augmented_text = aug_bert.augment(text)
        return augmented_text
    
    def process_bert_augmentation(self, data, column, act="substitute", p=0.3):
        augmented_data = data.copy()
        augmented_texts = []
        for text in tqdm(data[column], desc="Augmentation des données"):
            augmented_texts += self.bert_augmentation(text, act, p)
        augmented_data = augmented_data.drop(column, axis=1)
        augmented_data[column] = augmented_texts
        new_data = pd.concat([data, augmented_data]).reset_index(drop=True)
        return new_data