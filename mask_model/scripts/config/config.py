import os
import pickle

ASSET_NAME = "lstm_lstm_mask_model" # change manually
MODEL_NAME = "lstm_lstm_mask"
SEED_VALUE = 11
DATASET_NAME = "Covid-19_tweets"
FINE_TUNE_EMBEDDING_MODEL = False # False => "static", True => "non_static"
OPTIMIZER = "adam" # adam, adadelta (change manually)
LEARNING_RATE = 3e-5 # 1e-5, 3e-5, 5e-5, 10e-5
MINI_BATCH_SIZE = 50 # 30, 50
TRAIN_EPOCHS = 200
DROPOUT = 0.4
LIME_NO_OF_SAMPLES = 1000
HIDDEN_UNITS_CLASSIFIER = 256
HIDDEN_UNITS_MASK_EMBEDDER = 512

def load_configuration_parameters():

    config = {"asset_name":ASSET_NAME,
                "model_name":MODEL_NAME,
                "seed_value":SEED_VALUE,
                "dataset_name":DATASET_NAME,
                "fine_tune_embedding_model":FINE_TUNE_EMBEDDING_MODEL,
                "optimizer":OPTIMIZER,
                "learning_rate":LEARNING_RATE, 
                "mini_batch_size":MINI_BATCH_SIZE, 
                "train_epochs":TRAIN_EPOCHS,
                "dropout":DROPOUT,
                "lime_no_of_samples":LIME_NO_OF_SAMPLES,
                "hidden_units_classifier":HIDDEN_UNITS_CLASSIFIER,
                "hidden_units_mask_embedder":HIDDEN_UNITS_MASK_EMBEDDER,}
    return config

