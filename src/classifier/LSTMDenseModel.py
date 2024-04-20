from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, concatenate

class LSTMDenseModel:
    def __init__(self, lstm_units, embedding_dim, non_text_dim, max_sequence_length, nb_classes):
        self.max_sequence_length = 0
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.non_text_dim = non_text_dim
        self.max_sequence_length = max_sequence_length
        self.nb_classes = nb_classes
        self.model = self.build_model()

    def build_model(self):
            # On définit les couches d'entrées
            text_input = Input(shape=(self.max_sequence_length,))
            non_text_input = Input(shape=(self.non_text_dim,))
            

            # Couches d'embedding pour le texte
            embedding_layer = Embedding(input_dim=self.max_sequence_length, output_dim=self.embedding_dim)(text_input)

            # Couche LSTM pour le texte
            lstm_layer = LSTM(self.lstm_units)(embedding_layer)

            # Couche dense (fully-connected) pour les attributs numériques
            non_text_dense = Dense(32, activation='relu')(non_text_input)

            # Concatenation de la sortie de la couche LSTM avec la sortie de la couche Dense
            concatenated = concatenate([lstm_layer, non_text_dense])

            dense_layer = Dense(32, activation='relu')(concatenated)
            output = Dense(self.nb_classes, activation='softmax')(dense_layer) 

            # On définit le modèle
            model = Model(inputs=[text_input, non_text_input], outputs=output)
            return model

    def compile_model(self):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train_model(self, text_data, non_text_features, labels, epochs, batch_size):
        self.model.fit([text_data, non_text_features], labels, epochs=epochs, batch_size=batch_size, verbose=1)

    def test_model(self, text_data, non_text_features, labels):
        loss, accuracy = self.model.evaluate([text_data, non_text_features], labels, verbose=0)
        print(f"Test loss: {loss:.4f}")
        print(f"Test accuracy: {accuracy*100:.2f}%")