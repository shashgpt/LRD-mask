import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras import layers
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Model
# from tensorflow.python.ops.numpy_ops import np_config
# np_config.enable_numpy_behavior()


class FOL_rules(object):
    def __init__(self, classes, input, features):
        self.classes = classes
        self.input = input
        self.features = features

class FOL_A_but_B(FOL_rules):
    def __init__(self, classes, input, features):
        assert classes == 1
        super(FOL_A_but_B, self).__init__(classes, input, features)

    def log_distribution(self, w, batch_size, X=None, F=None):
        if F == None:
            X, F = self.input, self.features
        F_mask = F[:,0] #f_but_ind
        F_fea = F[:,1] #f_but_y_pred_p
        distr_y1 = tf.math.multiply(w, tf.math.multiply(F_mask, F_fea)) #y = 1 
        distr_y0 = tf.math.multiply(w, tf.math.multiply(F_mask, tf.math.subtract(1.0, F_fea))) #y = 0
        distr_y0 = tf.reshape(distr_y0, [batch_size, self.classes])
        distr_y1 = tf.reshape(distr_y1, [batch_size, self.classes])
        distr = tf.concat([distr_y0, distr_y1], axis=1)
        return distr

class Teacher_network(object):
    def __init__(self, batch_size, classes, rules, rules_lambda, teacher_regularizer):
        self.batch_size = batch_size
        self.classes = classes
        self.rules = rules
        self.rules_lambda = rules_lambda
        self.teacher_regularizer = teacher_regularizer

    def calc_rule_constraints(self, rules, rules_lambda, teacher_regularizer, batch_size, classes, new_data=None, new_rule_fea=None):
        if new_rule_fea==None:
            new_rule_fea = [None]*len(rules)
        distr_all = tf.zeros([batch_size, classes], dtype=tf.dtypes.float32)
        for i, rule in enumerate(rules):
            distr = rule.log_distribution(teacher_regularizer*rules_lambda[i], batch_size, new_data, new_rule_fea[i])
            distr_all = tf.math.add(distr_all, distr)
        distr_all = tf.math.add(distr_all, distr)
        distr_y0 = distr_all[:,0]
        distr_y0 = tf.reshape(distr_y0, [batch_size, 1])
        distr_y0_copies = tf.concat([tf.identity(distr_y0), tf.identity(distr_y0)], axis=1)
        distr_all = tf.math.subtract(distr_all, distr_y0_copies)
        distr_all = tf.math.maximum(tf.math.minimum(distr_all, tf.constant([60.])), tf.constant([-60.])) # truncate to avoid over-/under-flow
        distr_all = tf.math.exp(distr_all)
        return distr_all

    def teacher_output(self, student_output):
        distr = self.calc_rule_constraints(rules = self.rules, 
                                            rules_lambda = self.rules_lambda, 
                                            teacher_regularizer = self.teacher_regularizer, 
                                            batch_size = self.batch_size, 
                                            classes = self.classes)
        q_y_given_x = tf.math.multiply(student_output, distr)
        teacher_output = tf.math.divide(q_y_given_x, tf.reshape(tf.math.reduce_sum(q_y_given_x, axis=1), [-1, 1]))
        teacher_output = teacher_output[:,1]
        return teacher_output

class iteration_tracker(tf.keras.metrics.Metric):
    def __init__(self, name='iteration', **kwargs):
        super(iteration_tracker, self).__init__(name=name, **kwargs)
        self.iteration = self.add_weight(name='iteration', initializer='zeros')

    def update_state(self, curr_iter, sample_weight=None):
        self.iteration.assign_add(curr_iter)

    def result(self):
        return self.iteration

    def reset_states(self):
        self.iteration.assign(self.iteration)

class distillation_loss(tf.keras.metrics.Metric):
    def __init__(self, name='iteration', **kwargs):
        super(distillation_loss, self).__init__(name=name, **kwargs)
        self.distillation_loss = self.add_weight(name='distillation_loss', initializer='zeros')

    def update_state(self, distillation_loss, sample_weight=None):
        self.distillation_loss.assign(distillation_loss)

    def result(self):
        return self.distillation_loss

    def reset_states(self):
        self.distillation_loss.assign(0)

acc_tracker_per_epoch = tf.keras.metrics.BinaryAccuracy(name="accuracy")
iteration_tracker_metric = iteration_tracker()
distillation_loss_metric = distillation_loss()

class IKD(Model):

    def compute_mask(self, inputs, mask=None):
        # Also split the mask into 2 if it presents.
        if mask is None:
            return None
        return tf.split(mask, 2, axis=1)

    def train_step(self, data): # an iteration
        x,  y = data
        sentences = x[0]
        rule_features = x[1]
        sentiment_labels = y[0]
        rule_features_ind = y[1]

        with tf.GradientTape() as tape: # Forward propagation and loss calculation

            # # IKD from my understanding
            # y_pred = self(sentences, training=True)  #Forward pass
            # f_but_y_pred_p = self(rule_features, training=True)
            # distr = tf.math.multiply(f_but_y_pred_p, rule_features_ind, name=None) #check
            # distr = tf.math.maximum(tf.math.minimum(distr, tf.constant([60.])), tf.constant([-60.]))
            # multiply_but_exp = tf.math.exp(distr) #check
            # q_y_given_x = tf.math.multiply(y_pred, multiply_but_exp, name=None) #check
            # teacher = tf.math.divide(q_y_given_x, tf.reshape(tf.math.reduce_sum(q_y_given_x, axis=1), [-1, 1]))
            # loss_fn_data = tf.keras.losses.BinaryCrossentropy(from_logits=False)
            # loss_fn_rule = tf.keras.losses.BinaryCrossentropy(from_logits=False)
            # m = tf.math.multiply(self.iteration_tracker_metric.result(), 1./1408)
            # e = tf.math.pow(0.95, m)
            # max = tf.math.maximum(e, 0.0)
            # distillation_str = tf.math.subtract(1.0, max)
            # s1 = tf.math.subtract(1.0, distillation_str)
            # l1 = tf.math.multiply(loss_fn_data(sentiment_labels, y_pred), s1)
            # l2 = tf.math.multiply(loss_fn_rule(teacher, y_pred), distillation_str)
            # loss_value = tf.math.add(l1, l2)

            # IKD from authors code
            y_pred = self(sentences, training=True)
            f_but_y_pred_p = self(rule_features[0], training=True)
            f_yet_y_pred_p = self(rule_features[1], training=True)
            f_though_y_pred_p = self(rule_features[2], training=True)
            f_while_y_pred_p = self(rule_features[3], training=True)
            f_but_full = tf.concat([rule_features_ind[0], f_but_y_pred_p], axis=1)
            f_yet_full = tf.concat([rule_features_ind[1], f_yet_y_pred_p], axis=1)
            f_though_full = tf.concat([rule_features_ind[2], f_though_y_pred_p], axis=1)
            f_while_full = tf.concat([rule_features_ind[3], f_while_y_pred_p], axis=1)
            rules = [FOL_A_but_B(classes = 1, input = input, features = f_but_full), 
                    FOL_A_but_B(classes = 1, input = input, features = f_yet_full),
                    FOL_A_but_B(classes = 1, input = input, features = f_though_full),
                    FOL_A_but_B(classes = 1, input = input, features = f_while_full)]
            class_object = Teacher_network(batch_size = 50, classes = 1, rules = rules, rules_lambda = [1.0, 1.0, 1.0, 1.0], teacher_regularizer = 1.0)
            teacher = class_object.teacher_output(student_output = y_pred)
            loss_fn_data = tf.keras.losses.BinaryCrossentropy(from_logits=False)
            loss_fn_rule = tf.keras.losses.BinaryCrossentropy(from_logits=False)
            m = tf.math.multiply(iteration_tracker_metric.result(), 1./1408)
            e = tf.math.pow(0.95, m)
            max = tf.math.maximum(e, 0.0)
            distillation_str = tf.math.subtract(1.0, max)
            s1 = tf.math.subtract(1.0, distillation_str)
            l1 = tf.math.multiply(loss_fn_data(sentiment_labels, y_pred), s1)
            l2 = tf.math.multiply(loss_fn_rule(teacher, y_pred), distillation_str)
            loss_value = tf.math.add(l1, l2)
            # loss_value = loss_fn_data(sentiment_labels, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss_value, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        distillation_loss_metric.update_state(loss_value)
        acc_tracker_per_epoch.update_state(sentiment_labels, y_pred)
        iteration_tracker_metric.update_state(1.0)
        return {"loss": distillation_loss_metric.result(), "accuracy": acc_tracker_per_epoch.result(), "iteration": iteration_tracker_metric.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [distillation_loss_metric, acc_tracker_per_epoch, iteration_tracker_metric]
    
    def test_step(self, data):

        x, y = data
        sentences = x[0]
        rule_features = x[1]
        sentiment_labels = y[0]
        rule_features_ind = y[1]

        # Compute predictions
        y_pred = self(sentences, training=True)
        loss_fn_data = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        l1 = loss_fn_data(sentiment_labels, y_pred)

        # Compute our own metrics
        distillation_loss_metric.update_state(l1)
        acc_tracker_per_epoch.update_state(sentiment_labels, y_pred)
        return {"loss": distillation_loss_metric.result(), "accuracy": acc_tracker_per_epoch.result(), "iteration": iteration_tracker_metric.result()}

def lstm(config, word_vectors):

    # Input sentence (padded and tokenized)
    input_sentence = Input(shape=(None,), dtype="int64")
    
    # Word embeddings 
    out = layers.Embedding(word_vectors.shape[0], word_vectors.shape[1], 
                            embeddings_initializer=Constant(word_vectors), 
                            trainable=config["fine_tune_embedding_model"], 
                            mask_zero=True, 
                            name="word2vec")(input_sentence)

    # Classifier Layer
    out = layers.LSTM(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier")(out)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = IKD(inputs=[input_sentence], outputs=[out])
    if config["optimizer"] == "adam":    
        model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]))
    return model

def gru(config, word_vectors):

    # Input sentence (padded and tokenized)
    input_sentence = Input(shape=(None,), dtype="int64")
    
    # Word embeddings 
    out = layers.Embedding(word_vectors.shape[0], word_vectors.shape[1], 
                            embeddings_initializer=Constant(word_vectors), 
                            trainable=config["fine_tune_embedding_model"], 
                            mask_zero=True, 
                            name="word2vec")(input_sentence)

    # Classifier Layer
    out = layers.GRU(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier")(out)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = IKD(inputs=[input_sentence], outputs=[out])
    if config["optimizer"] == "adam":    
        model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]))
    return model

def rnn(config, word_vectors):

    # Input sentence (padded and tokenized)
    input_sentence = Input(shape=(None,), dtype="int64")
    
    # Word embeddings 
    out = layers.Embedding(word_vectors.shape[0], word_vectors.shape[1], 
                            embeddings_initializer=Constant(word_vectors), 
                            trainable=config["fine_tune_embedding_model"], 
                            mask_zero=True, 
                            name="word2vec")(input_sentence)

    # Classifier Layer
    out = layers.SimpleRNN(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier")(out)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = IKD(inputs=[input_sentence], outputs=[out])
    if config["optimizer"] == "adam":    
        model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]))
    return model

def bilstm(config, word_vectors):

    # Input sentence (padded and tokenized)
    input_sentence = Input(shape=(None,), dtype="int64")
    
    # Word embeddings 
    out = layers.Embedding(word_vectors.shape[0], word_vectors.shape[1], 
                            embeddings_initializer=Constant(word_vectors), 
                            trainable=config["fine_tune_embedding_model"], 
                            mask_zero=True, 
                            name="word2vec")(input_sentence)

    # Classifier Layer
    out = layers.Bidirectional(layers.LSTM(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier"))(out)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = IKD(inputs=[input_sentence], outputs=[out])
    if config["optimizer"] == "adam":    
        model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]))
    return model

def bigru(config, word_vectors):

    # Input sentence (padded and tokenized)
    input_sentence = Input(shape=(None,), dtype="int64")
    
    # Word embeddings 
    out = layers.Embedding(word_vectors.shape[0], word_vectors.shape[1], 
                            embeddings_initializer=Constant(word_vectors), 
                            trainable=config["fine_tune_embedding_model"], 
                            mask_zero=True, 
                            name="word2vec")(input_sentence)

    # Classifier Layer
    out = layers.Bidirectional(layers.GRU(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier"))(out)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = IKD(inputs=[input_sentence], outputs=[out])
    if config["optimizer"] == "adam":    
        model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]))
    return model

def birnn(config, word_vectors):

    # Input sentence (padded and tokenized)
    input_sentence = Input(shape=(None,), dtype="int64")
    
    # Word embeddings 
    out = layers.Embedding(word_vectors.shape[0], word_vectors.shape[1], 
                            embeddings_initializer=Constant(word_vectors), 
                            trainable=config["fine_tune_embedding_model"], 
                            mask_zero=True, 
                            name="word2vec")(input_sentence)

    # Classifier Layer
    out = layers.Bidirectional(layers.SimpleRNN(config["hidden_units_classifier"], dropout=config["dropout"], name="classifier"))(out)
    out = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # The model
    model = IKD(inputs=[input_sentence], outputs=[out])
    if config["optimizer"] == "adam":    
        model.compile(tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]))
    return model