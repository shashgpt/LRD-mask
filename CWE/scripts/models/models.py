import tensorflow as tf
from keras import Input
from keras import layers
from keras.initializers import Constant
from keras.models import Model
from keras.optimizers import adam

from keras import backend as K
from keras.engine.topology import Layer
import tensorflow_hub as hub

class ElmoEmbeddingLayer(Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable = True
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable, name="{}_module".format(self.name))
        self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1),
                       as_dict=True,
                       signature='default',
                       )['elmo']
        return result

    # def compute_mask(self, inputs, mask=None):
    #     return K.not_equal(inputs,'__PAD__')
    
    # def compute_mask(self, inputs, mask=None):
    #     return K.not_equal(inputs, '--PAD--')

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 64, self.dimensions)

# class ELMoEmbedding(Layer):
#     def __init__(self, idx2word, trainable, output_mode="default", **kwargs):
#         assert output_mode in ["default", "word_emb", "lstm_outputs1", "lstm_outputs2", "elmo"]
#         assert trainable in [True, False]
#         self.idx2word = idx2word
#         self.output_mode = output_mode
#         self.trainable = trainable
#         self.max_length = None
#         self.word_mapping = None
#         self.lookup_table = None
#         self.elmo_model = None
#         self.embedding = None
#         super(ELMoEmbedding, self).__init__(**kwargs)

#     def build(self, input_shape):
#         self.max_length = input_shape[1]
#         self.word_mapping = [x[1] for x in sorted(self.idx2word.items(), key=lambda x: x[0])]
#         self.lookup_table = tf.contrib.lookup.index_to_string_table_from_tensor(self.word_mapping, default_value="<UNK>")
#         self.lookup_table.init.run(session=K.get_session())
#         # self.elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=self.trainable)
#         self.elmo_model = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable, name="{}_module".format(self.name))
#         self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
#         super(ELMoEmbedding, self).build(input_shape)

#     def call(self, x):
#         x = tf.cast(x, dtype=tf.int64)
#         sequence_lengths = tf.cast(tf.count_nonzero(x, axis=1), dtype=tf.int32)
#         strings = self.lookup_table.lookup(x)
#         inputs = {
#             "tokens": strings,
#             "sequence_len": sequence_lengths
#         }
#         return self.elmo_model(inputs, signature="tokens", as_dict=True)[self.output_mode]

#     def compute_output_shape(self, input_shape):
#         if self.output_mode == "default":
#             return (input_shape[0], 1024)
#         if self.output_mode == "word_emb":
#             return (input_shape[0], self.max_length, 512)
#         if self.output_mode == "lstm_outputs1":
#             return (input_shape[0], self.max_length, 1024)
#         if self.output_mode == "lstm_outputs2":
#             return (input_shape[0], self.max_length, 1024)
#         if self.output_mode == "elmo":
#             return (input_shape[0], self.max_length, 1024)

#     def get_config(self):
#         config = {
#             'idx2word': self.idx2word,
#             'output_mode': self.output_mode 
#         }
#         return list(config.items())

# elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
# sess = tf.Session()
# K.set_session(sess)
# sess.run(tf.global_variables_initializer())
# sess.run(tf.tables_initializer())
# class MyLayer(Layer):

#     def __init__(self, output_dim, **kwargs):
#         self.output_dim = output_dim
#         super(MyLayer, self).__init__(**kwargs)

#     def build(self, input_shape):
#         # Create a trainable weight variable for this layer.

#         # These are the 3 trainable weights for word_embedding, lstm_output1 and lstm_output2
#         self.kernel1 = self.add_weight(name='kernel1',
#                                        shape=(3,),
#                                       initializer='uniform',
#                                       trainable=True)
#         # This is the bias weight
#         self.kernel2 = self.add_weight(name='kernel2',
#                                        shape=(),
#                                       initializer='uniform',
#                                       trainable=True)
#         super(MyLayer, self).build(input_shape)

#     def call(self, x):
#         # Get all the outputs of elmo_model
#         model =  elmo_model(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)
        
#         # Embedding activation output
#         activation1 = model["word_emb"]
        
#         # First LSTM layer output
#         activation2 = model["lstm_outputs1"]
        
#         # Second LSTM layer output
#         activation3 = model["lstm_outputs2"]

#         activation2 = tf.reduce_mean(activation2, axis=1)
#         activation3 = tf.reduce_mean(activation3, axis=1)
        
#         mul1 = tf.scalar_mul(self.kernel1[0], activation1)
#         mul2 = tf.scalar_mul(self.kernel1[1], activation2)
#         mul3 = tf.scalar_mul(self.kernel1[2], activation3)
        
#         sum_vector = tf.add(mul2, mul3)
        
#         return tf.scalar_mul(self.kernel2, sum_vector)

#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], self.output_dim)

def lstm(config, word_index):

    # Input sentence (padded and tokenized)
    input_sentence = layers.Input(shape=(1,), dtype=tf.string)
    # input_sentence = Input(shape=(config["max_seq_len"],), dtype=tf.int64)
    
    # Word embeddings
    # out = ELMoEmbedding(idx2word=word_index, output_mode="elmo", trainable=False)(input_sentence)
    out = ElmoEmbeddingLayer(name='ElmoEmbeddingLayer')(input_sentence)
    # out = MyLayer(output_dim=1024, trainable=True)(input_sentence)

    # Classifier Layer
    out = layers.LSTM(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier")(out)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = Model(inputs=[input_sentence], outputs=[out])
    if config["optimizer"] == "adam":    
        model.compile(adam(lr=config["learning_rate"]), loss=['binary_crossentropy'], metrics=['accuracy'])
    return model

def bilstm(config, word_index):

    # Input sentence (padded and tokenized)
    input_sentence = layers.Input(shape=(1,), dtype=tf.string)
    # input_sentence = Input(shape=(config["max_seq_len"],), dtype=tf.int64)
    
    # Word embeddings
    # out = ELMoEmbedding(idx2word=word_index, output_mode="elmo", trainable=False)(input_sentence)
    out = ElmoEmbeddingLayer(name='ElmoEmbeddingLayer')(input_sentence)
    # out = MyLayer(output_dim=1024, trainable=True)(input_sentence)

    # Classifier Layer
    out_forward = layers.LSTM(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier_1")(out)
    out_backward = layers.LSTM(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier_2", go_backwards=True)(out)
    out = layers.merge.Concatenate(axis=1)([out_forward, out_backward])
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = Model(inputs=[input_sentence], outputs=[out])
    if config["optimizer"] == "adam":    
        model.compile(adam(lr=config["learning_rate"]), loss=['binary_crossentropy'], metrics=['accuracy'])
    return model

def gru(config, word_index):

    # Input sentence (padded and tokenized)
    input_sentence = layers.Input(shape=(1,), dtype=tf.string)
    # input_sentence = Input(shape=(config["max_seq_len"],), dtype=tf.int64)
    
    # Word embeddings
    # out = ELMoEmbedding(idx2word=word_index, output_mode="elmo", trainable=False)(input_sentence)
    out = ElmoEmbeddingLayer(name='ElmoEmbeddingLayer')(input_sentence)
    # out = MyLayer(output_dim=1024, trainable=True)(input_sentence)

    # Classifier Layer
    out = layers.GRU(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier")(out)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = Model(inputs=[input_sentence], outputs=[out])
    if config["optimizer"] == "adam":    
        model.compile(adam(lr=config["learning_rate"]), loss=['binary_crossentropy'], metrics=['accuracy'])
    return model

def bigru(config, word_index):

    # Input sentence (padded and tokenized)
    input_sentence = layers.Input(shape=(1,), dtype=tf.string)
    # input_sentence = Input(shape=(config["max_seq_len"],), dtype=tf.int64)
    
    # Word embeddings
    # out = ELMoEmbedding(idx2word=word_index, output_mode="elmo", trainable=False)(input_sentence)
    out = ElmoEmbeddingLayer(name='ElmoEmbeddingLayer')(input_sentence)
    # out = MyLayer(output_dim=1024, trainable=True)(input_sentence)

    # Classifier Layer
    out_forward = layers.GRU(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier_1")(out)
    out_backward = layers.GRU(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier_2", go_backwards=True)(out)
    out = layers.merge.Concatenate(axis=1)([out_forward, out_backward])
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = Model(inputs=[input_sentence], outputs=[out])
    if config["optimizer"] == "adam":    
        model.compile(adam(lr=config["learning_rate"]), loss=['binary_crossentropy'], metrics=['accuracy'])
    return model

def rnn(config, word_index):

    # Input sentence (padded and tokenized)
    input_sentence = layers.Input(shape=(1,), dtype=tf.string)
    # input_sentence = Input(shape=(config["max_seq_len"],), dtype=tf.int64)
    
    # Word embeddings
    # out = ELMoEmbedding(idx2word=word_index, output_mode="elmo", trainable=False)(input_sentence)
    out = ElmoEmbeddingLayer(name='ElmoEmbeddingLayer')(input_sentence)
    # out = MyLayer(output_dim=1024, trainable=True)(input_sentence)

    # Classifier Layer
    out = layers.SimpleRNN(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier")(out)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = Model(inputs=[input_sentence], outputs=[out])
    if config["optimizer"] == "adam":    
        model.compile(adam(lr=config["learning_rate"]), loss=['binary_crossentropy'], metrics=['accuracy'])
    return model

def birnn(config, word_index):

    # Input sentence (padded and tokenized)
    input_sentence = layers.Input(shape=(1,), dtype=tf.string)
    # input_sentence = Input(shape=(config["max_seq_len"],), dtype=tf.int64)
    
    # Word embeddings
    # out = ELMoEmbedding(idx2word=word_index, output_mode="elmo", trainable=False)(input_sentence)
    out = ElmoEmbeddingLayer(name='ElmoEmbeddingLayer')(input_sentence)
    # out = MyLayer(output_dim=1024, trainable=True)(input_sentence)

    # Classifier Layer
    out_forward = layers.SimpleRNN(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier_1")(out)
    out_backward = layers.SimpleRNN(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier_2", go_backwards=True)(out)
    out = layers.merge.Concatenate(axis=1)([out_forward, out_backward])
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = Model(inputs=[input_sentence], outputs=[out])
    if config["optimizer"] == "adam":    
        model.compile(adam(lr=config["learning_rate"]), loss=['binary_crossentropy'], metrics=['accuracy'])
    return model