import os
import pickle
import numpy as np
import tensorflow as tf

class Evaluation(object):
    def __init__(self, config, word_index):
        self.config = config
        self.word_index = word_index
        self.vocab = [key for key in self.word_index.keys()]
        self.vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(standardize=None, split='whitespace', vocabulary=self.vocab)
    
    def vectorize(self, sentences):
        """
        tokenize each sentence in dataset as sentence.split()
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
            elif rule_label!=rule or contrast!=1:
                rule_conjuncts.append('')
                rule_label_ind.append(0)
        return rule_conjuncts, rule_label_ind
    
    def remove_extra_samples(self, sample):
        sample = sample[:(sample.shape[0]-sample.shape[0]%self.config["mini_batch_size"])]
        return sample

    def evaluate_model(self, model, test_datasets):

        # Results to be created after evaluation
        results = {'sentence':[], 
                    'sentiment_label':[],  
                    'rule_label':[],
                    'contrast':[],
                    'sentiment_probability_output':[], 
                    'sentiment_prediction_output':[]}

        test_dataset = test_datasets["test_dataset"]
        test_sentences = self.vectorize(test_dataset["sentence"])
        test_sentiment_labels = np.array(test_dataset["sentiment_label"])
        test_sentences_but_features, test_sentences_but_features_ind = self.rule_conjunct_extraction(test_dataset, rule=1)
        test_sentences_yet_features, test_sentences_yet_features_ind = self.rule_conjunct_extraction(test_dataset, rule=2)
        test_sentences_though_features, test_sentences_though_features_ind = self.rule_conjunct_extraction(test_dataset, rule=3)
        test_sentences_while_features, test_sentences_while_features_ind = self.rule_conjunct_extraction(test_dataset, rule=4)

        test_sentences_but_features = self.vectorize(test_sentences_but_features)
        test_sentences_yet_features = self.vectorize(test_sentences_yet_features)
        test_sentences_though_features = self.vectorize(test_sentences_though_features)
        test_sentences_while_features = self.vectorize(test_sentences_while_features)

        test_sentences_but_features_ind = np.array(test_sentences_but_features_ind).astype(np.float32)
        test_sentences_but_features_ind = test_sentences_but_features_ind.reshape(test_sentences_but_features_ind.shape[0], 1)
        test_sentences_yet_features_ind = np.array(test_sentences_yet_features_ind).astype(np.float32)
        test_sentences_yet_features_ind = test_sentences_yet_features_ind.reshape(test_sentences_yet_features_ind.shape[0], 1)
        test_sentences_though_features_ind = np.array(test_sentences_though_features_ind).astype(np.float32)
        test_sentences_though_features_ind = test_sentences_though_features_ind.reshape(test_sentences_though_features_ind.shape[0], 1)
        test_sentences_while_features_ind = np.array(test_sentences_while_features_ind).astype(np.float32)
        test_sentences_while_features_ind = test_sentences_while_features_ind.reshape(test_sentences_while_features_ind.shape[0], 1)

        test_sentences = self.remove_extra_samples(test_sentences)
        test_sentiment_labels = self.remove_extra_samples(test_sentiment_labels)
        test_sentences_but_features = self.remove_extra_samples(test_sentences_but_features)
        test_sentences_yet_features = self.remove_extra_samples(test_sentences_yet_features)
        test_sentences_though_features = self.remove_extra_samples(test_sentences_though_features)
        test_sentences_while_features = self.remove_extra_samples(test_sentences_while_features)
        test_sentences_but_features_ind = self.remove_extra_samples(test_sentences_but_features_ind)
        test_sentences_yet_features_ind = self.remove_extra_samples(test_sentences_yet_features_ind)
        test_sentences_though_features_ind = self.remove_extra_samples(test_sentences_though_features_ind)
        test_sentences_while_features_ind = self.remove_extra_samples(test_sentences_while_features_ind)
        dataset = ([test_sentences, [test_sentences_but_features, test_sentences_yet_features, test_sentences_though_features, test_sentences_while_features]], 
                            [test_sentiment_labels, [test_sentences_but_features_ind, test_sentences_yet_features_ind, test_sentences_though_features_ind, test_sentences_while_features_ind]])

        # Evaluation and predictions
        # evaluations = model.evaluate(x=dataset[0], y=dataset[1])
        # print(evaluations)
        predictions = model.predict(self.vectorize(test_dataset["sentence"]))

        for index, sentence in enumerate(test_dataset["sentence"]):
            results['sentence'].append(test_dataset['sentence'][index])
            results['sentiment_label'].append(test_dataset['sentiment_label'][index])
            results['rule_label'].append(test_dataset['rule_label'][index])
            results['contrast'].append(test_dataset['contrast'][index])
        for prediction in predictions:
            results['sentiment_probability_output'].append(prediction)
            prediction = np.rint(prediction)
            results['sentiment_prediction_output'].append(prediction[0])

        # Save the results
        if not os.path.exists("assets/results/"):
            os.makedirs("assets/results/")
        with open("assets/results/"+self.config["asset_name"]+".pickle", 'wb') as handle:
            pickle.dump(results, handle)