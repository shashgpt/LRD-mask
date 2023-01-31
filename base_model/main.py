import logging
import os
import pickle
import numpy as np
import random
import tensorflow as tf
import warnings
import pandas as pd
from tensorflow.keras.utils import plot_model
# from tokenizers import BertWordPieceTokenizer
# from keras_bert import get_base_dict, get_model, compile_model, gen_batch_inputs

# Scrips imports
from scripts.config.config import load_configuration_parameters
from scripts.dataset_processing.preprocess_dataset import Preprocess_dataset
from scripts.dataset_processing.word_vectors import Word_vectors
from scripts.dataset_processing.dataset_division import Dataset_division
from scripts.models.models import *
from scripts.train_models.train import Train
from scripts.evaluate_models.evaluation import Evaluation
from scripts.explanations.shap_explanations import Shap_explanations
from scripts.explanations.lime_explanations import Lime_explanations

# Change the code execution directory to current directory
os.chdir(os.getcwd())

# Disable warnings
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

if __name__=='__main__':

    # Gather configuration parameters
    config = load_configuration_parameters()
    print("\n"+config["asset_name"])

    # Set seed value
    os.environ['PYTHONHASHSEED']=str(config["seed_value"])
    random.seed(config["seed_value"])
    np.random.seed(config["seed_value"])
    tf.random.set_seed(config["seed_value"])

    # Create input data for model
    print("\nCreating input data")
    raw_dataset = pickle.load(open("datasets/"+config["dataset_name"]+"/raw_dataset/dataset.pickle", "rb"))
    raw_dataset = pd.DataFrame(raw_dataset)
    preprocessed_dataset = Preprocess_dataset(config).preprocess_covid_tweets(raw_dataset)
    word_vectors, word_index = Word_vectors(config).create_word_vectors(preprocessed_dataset)
    train_dataset, val_datasets, test_datasets = Dataset_division(config).train_val_test_split(preprocessed_dataset)
   
    # Create model
    print("\nBuilding model")
    if config["model_name"] == "rnn":
        model = rnn(config, word_vectors)
    elif config["model_name"] == "birnn":
        model = birnn(config, word_vectors)
    elif config["model_name"] == "gru":
        model = gru(config, word_vectors)
    elif config["model_name"] == "bigru":
        model = bigru(config, word_vectors)
    elif config["model_name"] == "lstm":
        model = lstm(config, word_vectors)
    elif config["model_name"] == "bilstm":
        model = bilstm(config, word_vectors)
    model.summary(line_length = 150)
    if not os.path.exists("assets/computation_graphs"):
        os.makedirs("assets/computation_graphs")
    plot_model(model, show_shapes = True, to_file = "assets/computation_graphs/"+config["asset_name"]+".png")

    # # Train model
    # print("\nTraining")
    # Train(config, word_index).train_model(model, train_dataset, val_datasets, test_datasets)

    # Load trained model
    model.load_weights("assets/trained_models/"+config["asset_name"]+".h5")

    # # Test model
    # print("\nEvaluation")
    # Evaluation(config, word_index).evaluate_model(model, test_datasets)

    # # LIME explanations
    # print("\nLIME explanations")
    # Lime_explanations(config, model, word_index).create_lime_explanations()

    # Shap explanations
    print("\nSHAP explanations")
    Shap_explanations(config, model, word_index).create_shap_explanations(train_dataset)

    # Save the configuration parameters for this run (marks the creation of an asset)
    if "test" not in config["asset_name"]: 
        if not os.path.exists("assets/configurations/"):
            os.makedirs("assets/configurations/")
        with open("assets/configurations/"+config["asset_name"]+".pickle", 'wb') as handle:
            pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)