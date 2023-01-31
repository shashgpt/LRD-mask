import logging
import os
import pickle
import numpy as np
import random
import tensorflow as tf
import warnings
import pandas as pd
from tensorflow.keras.utils import plot_model

# Scrips imports
from scripts.config.config import load_configuration_parameters
from scripts.dataset_processing.preprocess_dataset import Preprocess_dataset
from scripts.dataset_processing.word_vectors import Word_vectors
from scripts.dataset_processing.dataset_division import Dataset_division
from scripts.models.models import *
from scripts.train_models.train import Train
from scripts.evaluate_models.evaluation import Evaluation
# from scripts.explanations.shap_explanations import Shap_explanations
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
    if config["model_name"] == "rnn_bilstm_mask":
        model = rnn_bilstm_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "birnn_bilstm_mask":
        model = birnn_bilstm_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "gru_bilstm_mask":
        model = gru_bilstm_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "bigru_bilstm_mask":
        model = bigru_bilstm_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "lstm_bilstm_mask":
        model = lstm_bilstm_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "bilstm_bilstm_mask":
        model = bilstm_bilstm_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)



    elif config["model_name"] == "rnn_bigru_mask":
        model = rnn_bigru_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "birnn_bigru_mask":
        model = birnn_bigru_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "gru_bigru_mask":
        model = gru_bigru_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "bigru_bigru_mask":
        model = bigru_bigru_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "lstm_bigru_mask":
        model = lstm_bigru_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "bilstm_bigru_mask":
        model = bilstm_bigru_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    

    elif config["model_name"] == "rnn_lstm_mask":
        model = rnn_lstm_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "birnn_lstm_mask":
        model = birnn_lstm_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "gru_lstm_mask":
        model = gru_lstm_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "bigru_lstm_mask":
        model = bigru_lstm_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "lstm_lstm_mask":
        model = lstm_lstm_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "bilstm_lstm_mask":
        model = bilstm_lstm_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)


    elif config["model_name"] == "rnn_rnn_mask":
        model = rnn_rnn_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "birnn_rnn_mask":
        model = birnn_rnn_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "gru_rnn_mask":
        model = gru_rnn_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "bigru_rnn_mask":
        model = bigru_rnn_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "lstm_rnn_mask":
        model = lstm_rnn_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "bilstm_rnn_mask":
        model = bilstm_rnn_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    
    elif config["model_name"] == "rnn_gru_mask":
        model = rnn_gru_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "birnn_gru_mask":
        model = birnn_gru_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "gru_gru_mask":
        model = gru_gru_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "bigru_gru_mask":
        model = bigru_gru_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "lstm_gru_mask":
        model = lstm_gru_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "bilstm_gru_mask":
        model = bilstm_gru_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    
    elif config["model_name"] == "rnn_birnn_mask":
        model = rnn_birnn_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "birnn_birnn_mask":
        model = birnn_birnn_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "gru_birnn_mask":
        model = gru_birnn_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "bigru_birnn_mask":
        model = bigru_birnn_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "lstm_birnn_mask":
        model = lstm_birnn_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    elif config["model_name"] == "bilstm_birnn_mask":
        model = bilstm_birnn_mask(config, word_vectors)
        # model.load_weights("assets/trained_models/"+config["asset_name"]+".h5", by_name=True)
    
    model.summary(line_length=150)
    if not os.path.exists("assets/computation_graphs"):
        os.makedirs("assets/computation_graphs")
    plot_model(model, show_shapes=True, to_file="assets/computation_graphs/"+config["asset_name"]+".png")

    # Train model
    print("\nTraining")
    model = Train(config, word_index).train_model(model, train_dataset, val_datasets, test_datasets)

    # Load trained model
    model.load_weights("assets/trained_models/"+config["asset_name"]+".h5")

    # Test model
    print("\nEvaluation")
    Evaluation(config, word_index).evaluate_model(model, test_datasets)

    # LIME explanations
    print("\nLIME explanations")
    Lime_explanations(config, model, word_index).create_lime_explanations()

    # Save the configuration parameters for this run (marks the creation of an asset)
    if "test" not in config["asset_name"]: 
        if not os.path.exists("assets/configurations/"):
            os.makedirs("assets/configurations/")
        with open("assets/configurations/"+config["asset_name"]+".pickle", 'wb') as handle:
            pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)