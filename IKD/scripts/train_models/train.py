import os
import shutil
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
from sklearn.utils import shuffle
import pandas as pd
from tensorflow.keras import Input
from tensorflow.keras import layers
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Model

from scripts.train_models.additional_validation_sets import AdditionalValidationSets

class Train(object):
    def __init__(self, config, word_index):
        self.config = config
        self.word_index = word_index
        self.vocab = [key for key in self.word_index.keys()]
        self.vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(standardize=None, split='whitespace', vocabulary=self.vocab)
    
    def vectorize(self, sentences):
        """
        tokenize each preprocessed sentence in dataset as sentence.split()
        encode each tokenized sentence as per vocabulary
        right pad each encoded tokenized sentence with 0 upto max_tokenized_sentence_len the dataset 
        """
        return self.vectorize_layer(np.array(sentences)).numpy()

    def rule_conjunct_extraction(self, dataset, rule):
        """
        Extracts the rule_conjuncts from sentences containing the logic rule corresponding to rule_keyword
        """
        rule_conjuncts = []
        rule_label_ind = []
        for index, sentence in enumerate(list(dataset['sentence'])):
            tokenized_sentence = sentence.split()
            rule_label = dataset['rule_label'][index]
            contrast = dataset['contrast'][index]
            if rule_label == rule and contrast==1:
                if rule_label == 1:
                    b_part_tokenized_sentence = tokenized_sentence[tokenized_sentence.index("but")+1:]
                    b_part_sentence = ' '.join(b_part_tokenized_sentence)
                    rule_conjuncts.append(b_part_sentence)
                    rule_label_ind.append(1)
                elif rule_label == 2:
                    b_part_tokenized_sentence = tokenized_sentence[tokenized_sentence.index("yet")+1:]
                    b_part_sentence = ' '.join(b_part_tokenized_sentence)
                    rule_conjuncts.append(b_part_sentence)
                    rule_label_ind.append(1)
                elif rule_label == 3:
                    a_part_tokenized_sentence = tokenized_sentence[:tokenized_sentence.index("though")]
                    a_part_sentence = ' '.join(a_part_tokenized_sentence)
                    rule_conjuncts.append(a_part_sentence)
                    rule_label_ind.append(1)
                elif rule_label == 4:
                    a_part_tokenized_sentence = tokenized_sentence[:tokenized_sentence.index("while")]
                    a_part_sentence = ' '.join(a_part_tokenized_sentence)
                    rule_conjuncts.append(a_part_sentence)
                    rule_label_ind.append(1)
            else:
                rule_conjuncts.append('')
                rule_label_ind.append(0)
        return rule_conjuncts, rule_label_ind
    
    def remove_extra_samples(self, sample):
        sample = sample[:(sample.shape[0]-sample.shape[0]%self.config["mini_batch_size"])]
        return sample
    
    def train_model(self, model, train_dataset, val_datasets, test_datasets):

        # Make paths
        if not os.path.exists("assets/training_log/"):
            os.makedirs("assets/training_log/")

        # Create train and val datasets
        train_sentences = self.vectorize(train_dataset["sentence"])
        train_sentiment_labels = np.array(train_dataset["sentiment_label"])
        val_sentences = self.vectorize(val_datasets["val_dataset"]["sentence"])
        val_sentiment_labels = np.array(val_datasets["val_dataset"]["sentiment_label"])

        # Create rule features
        train_sentences_but_features, train_sentences_but_features_ind = self.rule_conjunct_extraction(train_dataset, rule=1)
        train_sentences_yet_features, train_sentences_yet_features_ind = self.rule_conjunct_extraction(train_dataset, rule=2)
        train_sentences_though_features, train_sentences_though_features_ind = self.rule_conjunct_extraction(train_dataset, rule=3)
        train_sentences_while_features, train_sentences_while_features_ind = self.rule_conjunct_extraction(train_dataset, rule=4)

        train_sentences_but_features = self.vectorize(train_sentences_but_features)
        train_sentences_but_features_ind = np.array(train_sentences_but_features_ind).astype(np.float32)
        train_sentences_but_features_ind = train_sentences_but_features_ind.reshape(train_sentences_but_features_ind.shape[0], 1)

        train_sentences_yet_features = self.vectorize(train_sentences_yet_features)
        train_sentences_yet_features_ind = np.array(train_sentences_yet_features_ind).astype(np.float32)
        train_sentences_yet_features_ind = train_sentences_yet_features_ind.reshape(train_sentences_yet_features_ind.shape[0], 1)

        train_sentences_though_features = self.vectorize(train_sentences_though_features)
        train_sentences_though_features_ind = np.array(train_sentences_though_features_ind).astype(np.float32)
        train_sentences_though_features_ind = train_sentences_though_features_ind.reshape(train_sentences_though_features_ind.shape[0], 1)

        train_sentences_while_features = self.vectorize(train_sentences_while_features)
        train_sentences_while_features_ind = np.array(train_sentences_while_features_ind).astype(np.float32)
        train_sentences_while_features_ind = train_sentences_while_features_ind.reshape(train_sentences_while_features_ind.shape[0], 1)

        val_sentences_but_features, val_sentences_but_features_ind = self.rule_conjunct_extraction(val_datasets["val_dataset"], rule=1)
        val_sentences_yet_features, val_sentences_yet_features_ind = self.rule_conjunct_extraction(val_datasets["val_dataset"], rule=2)
        val_sentences_though_features, val_sentences_though_features_ind = self.rule_conjunct_extraction(val_datasets["val_dataset"], rule=3)
        val_sentences_while_features, val_sentences_while_features_ind = self.rule_conjunct_extraction(val_datasets["val_dataset"], rule=4)

        val_sentences_but_features = self.vectorize(val_sentences_but_features)
        val_sentences_but_features_ind = np.array(val_sentences_but_features_ind).astype(np.float32)
        val_sentences_but_features_ind = val_sentences_but_features_ind.reshape(val_sentences_but_features_ind.shape[0], 1)

        val_sentences_yet_features = self.vectorize(val_sentences_yet_features)
        val_sentences_yet_features_ind = np.array(val_sentences_yet_features_ind).astype(np.float32)
        val_sentences_yet_features_ind = val_sentences_yet_features_ind.reshape(val_sentences_yet_features_ind.shape[0], 1)

        val_sentences_though_features = self.vectorize(val_sentences_though_features)
        val_sentences_though_features_ind = np.array(val_sentences_though_features_ind).astype(np.float32)
        val_sentences_though_features_ind = val_sentences_though_features_ind.reshape(val_sentences_though_features_ind.shape[0], 1)

        val_sentences_while_features = self.vectorize(val_sentences_while_features)
        val_sentences_while_features_ind = np.array(val_sentences_while_features_ind).astype(np.float32)
        val_sentences_while_features_ind = val_sentences_while_features_ind.reshape(val_sentences_while_features_ind.shape[0], 1)

        train_sentences = self.remove_extra_samples(train_sentences)
        train_sentiment_labels = self.remove_extra_samples(train_sentiment_labels)
        train_sentences_but_features = self.remove_extra_samples(train_sentences_but_features)
        train_sentences_yet_features = self.remove_extra_samples(train_sentences_yet_features)
        train_sentences_though_features = self.remove_extra_samples(train_sentences_though_features)
        train_sentences_while_features = self.remove_extra_samples(train_sentences_while_features)
        train_sentences_but_features_ind = self.remove_extra_samples(train_sentences_but_features_ind)
        train_sentences_yet_features_ind = self.remove_extra_samples(train_sentences_yet_features_ind)
        train_sentences_though_features_ind = self.remove_extra_samples(train_sentences_though_features_ind)
        train_sentences_while_features_ind = self.remove_extra_samples(train_sentences_while_features_ind)
        train_dataset = ([train_sentences, [train_sentences_but_features, train_sentences_yet_features, train_sentences_though_features, train_sentences_while_features]], 
                        [train_sentiment_labels, [train_sentences_but_features_ind, train_sentences_yet_features_ind, train_sentences_though_features_ind, train_sentences_while_features_ind]])
        
        val_sentences = self.remove_extra_samples(val_sentences)
        val_sentiment_labels = self.remove_extra_samples(val_sentiment_labels)
        val_sentences_but_features = self.remove_extra_samples(val_sentences_but_features)
        val_sentences_yet_features = self.remove_extra_samples(val_sentences_yet_features)
        val_sentences_though_features = self.remove_extra_samples(val_sentences_though_features)
        val_sentences_while_features = self.remove_extra_samples(val_sentences_while_features)
        val_sentences_but_features_ind = self.remove_extra_samples(val_sentences_but_features_ind)
        val_sentences_yet_features_ind = self.remove_extra_samples(val_sentences_yet_features_ind)
        val_sentences_though_features_ind = self.remove_extra_samples(val_sentences_though_features_ind)
        val_sentences_while_features_ind = self.remove_extra_samples(val_sentences_while_features_ind)
        val_dataset = ([val_sentences, [val_sentences_but_features, val_sentences_yet_features, val_sentences_though_features, val_sentences_while_features]], 
                        [val_sentiment_labels, [val_sentences_but_features_ind, val_sentences_yet_features_ind, val_sentences_though_features_ind, val_sentences_while_features_ind]])

        # Additional datasets for calculating metrics per epoch
        additional_validation_datasets = []
        for key, value in test_datasets.items():
            sentences = self.vectorize(test_datasets[key]["sentence"])
            sentiment_labels = np.array(test_datasets[key]["sentiment_label"])

            sentences_but_features, sentences_but_features_ind = self.rule_conjunct_extraction(test_datasets[key], rule=1)
            sentences_yet_features, sentences_yet_features_ind = self.rule_conjunct_extraction(test_datasets[key], rule=2)
            sentences_though_features, sentences_though_features_ind = self.rule_conjunct_extraction(test_datasets[key], rule=3)
            sentences_while_features, sentences_while_features_ind = self.rule_conjunct_extraction(test_datasets[key], rule=4)

            sentences_but_features = self.vectorize(sentences_but_features)
            sentences_yet_features = self.vectorize(sentences_yet_features)
            sentences_though_features = self.vectorize(sentences_though_features)
            sentences_while_features = self.vectorize(sentences_while_features)

            sentences_but_features_ind = np.array(sentences_but_features_ind).astype(np.float32)
            sentences_but_features_ind = sentences_but_features_ind.reshape(sentences_but_features_ind.shape[0], 1)
            sentences_yet_features_ind = np.array(sentences_yet_features_ind).astype(np.float32)
            sentences_yet_features_ind = sentences_yet_features_ind.reshape(sentences_yet_features_ind.shape[0], 1)
            sentences_though_features_ind = np.array(sentences_though_features_ind).astype(np.float32)
            sentences_though_features_ind = sentences_though_features_ind.reshape(sentences_though_features_ind.shape[0], 1)
            sentences_while_features_ind = np.array(sentences_while_features_ind).astype(np.float32)
            sentences_while_features_ind = sentences_while_features_ind.reshape(sentences_while_features_ind.shape[0], 1)

            sentences = self.remove_extra_samples(sentences)
            sentiment_labels = self.remove_extra_samples(sentiment_labels)
            sentences_but_features = self.remove_extra_samples(sentences_but_features)
            sentences_yet_features = self.remove_extra_samples(sentences_yet_features)
            sentences_though_features = self.remove_extra_samples(sentences_though_features)
            sentences_while_features = self.remove_extra_samples(sentences_while_features)
            sentences_but_features_ind = self.remove_extra_samples(sentences_but_features_ind)
            sentences_yet_features_ind = self.remove_extra_samples(sentences_yet_features_ind)
            sentences_though_features_ind = self.remove_extra_samples(sentences_though_features_ind)
            sentences_while_features_ind = self.remove_extra_samples(sentences_while_features_ind)
            dataset = ([sentences, [sentences_but_features, sentences_yet_features, sentences_though_features, sentences_while_features]], 
                            [sentiment_labels, [sentences_but_features_ind, sentences_yet_features_ind, sentences_though_features_ind, sentences_while_features_ind]], key)
            additional_validation_datasets.append(dataset)

        # Define callbacks
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',          # 1. Calculate val_loss_1 
                                                                    min_delta = 0,                  # 2. Check val_losses for next 10 epochs 
                                                                    patience=10,                    # 3. Stop training if none of the val_losses are lower than val_loss_1
                                                                    verbose=0,                      # 4. Get the trained weights corresponding to val_loss_1
                                                                    mode="max",
                                                                    baseline=None, 
                                                                    restore_best_weights=True)
        my_callbacks = [early_stopping_callback, AdditionalValidationSets(additional_validation_datasets, self.config)]
        model.fit(x=train_dataset[0],
                    y=train_dataset[1],
                    epochs=self.config["train_epochs"], 
                    batch_size=self.config["mini_batch_size"], 
                    validation_data=val_dataset,
                    callbacks=my_callbacks,
                    shuffle=False)

        if not os.path.exists("assets/trained_models/"):
            os.makedirs("assets/trained_models/")
        model.save_weights("assets/trained_models/"+self.config["asset_name"]+".h5")

        return model