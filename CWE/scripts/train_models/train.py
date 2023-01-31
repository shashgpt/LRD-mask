import os
from random import shuffle
import shutil
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras

from scripts.train_models.additional_validation_sets import AdditionalValidationSets

class Train(object):
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
    
    def encoding_and_padding(self, sentences):
        encoded_tokenized_sentences = self.tokenizer.texts_to_sequences(sentences)
        padded_encoded_tokenized_sentences = keras.preprocessing.sequence.pad_sequences(encoded_tokenized_sentences, maxlen=self.config["max_seq_len"], value=0, padding='post', truncating='post')
        return padded_encoded_tokenized_sentences

    def train_model(self, model, train_dataset, val_datasets, test_datasets):
        
        # Make paths
        if not os.path.exists("assets/training_log/"):
            os.makedirs("assets/training_log/")

        # Create train and val datasets
        # train_sentences = self.encoding_and_padding(train_dataset["sentence"])
        train_sentences = np.array(train_dataset["sentence"])
        train_sentiment_labels = np.array(train_dataset["sentiment_label"])

        # val_sentences = self.encoding_and_padding(val_datasets["val_dataset"]["sentence"])
        val_sentences = np.array(val_datasets["val_dataset"]["sentence"])
        val_sentiment_labels = np.array(val_datasets["val_dataset"]["sentiment_label"])

        train_dataset = (train_sentences, train_sentiment_labels) 
        val_dataset = (val_sentences, val_sentiment_labels)

        # Additional datasets for calculating metrics per epoch
        additional_validation_datasets = []
        for key, value in test_datasets.items():
            # sentences = self.encoding_and_padding(test_datasets[key]["sentence"])
            sentences = np.array(test_datasets[key]["sentence"])
            sentiment_labels = np.array(test_datasets[key]["sentiment_label"])
            dataset = (sentences, sentiment_labels, key)
            additional_validation_datasets.append(dataset)

        # Define callbacks
        if not os.path.exists("assets/trained_models/"):
            os.makedirs("assets/trained_models/")
        
        # Stop the training if val_acc doesn't improve for 5 continuous epochs
        early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=7, verbose=0, mode='max')

        # Save the model corresponding to best val_acc during training
        model_checkpoint = keras.callbacks.ModelCheckpoint("assets/trained_models/"+self.config["asset_name"]+".h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='max', period=1)

        my_callbacks = [early_stopping_callback, model_checkpoint, AdditionalValidationSets(additional_validation_datasets, self.config)]
        model.fit(x=train_dataset[0], 
                    y=train_dataset[1], 
                    epochs=self.config["train_epochs"], 
                    batch_size=self.config["mini_batch_size"], 
                    validation_data=val_dataset, 
                    callbacks=my_callbacks,
                    shuffle=False)

        return model
    