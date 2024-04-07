import random
from tqdm import tqdm

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

    def augment(self, eda, data, N, alpha):
        augmented_data = data.copy()

        indexes = random.sample(range(len(data)), N)
        for index in tqdm(indexes, desc = "Augmentation des données"):
            row = data.loc[index]
            augmented_row = self.apply_eda(eda, row, alpha)

            augmented_data.loc[len(augmented_data)] = augmented_row

        return augmented_data