#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import tensorflow.contrib.summary  # Needed to allow importing summary operations
import warnings
warnings.simplefilter("ignore", DeprecationWarning)


class Network:

    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(
            graph=graph,
            config=tf.ConfigProto(
                inter_op_parallelism_threads=threads,
                intra_op_parallelism_threads=threads))

    def construct(self, params, logdir):
        self.NUM_OF_FEATURES = params['num_of_features']
        self.LABELS = 2
        with self.session.graph.as_default():
            # Inputs
            self.array = tf.placeholder(
                tf.float32, [None, self.NUM_OF_FEATURES], name='array')
            self.labels = tf.placeholder(tf.int64, [None], name='labels')
            self.is_training = tf.placeholder(tf.bool, [], name='is_training')

            # Computation
            flattened_array = tf.layers.flatten(self.array, name='flatten')
            layers = [flattened_array]
            for i in range(params['layers']):
                layers.append(
                    tf.layers.dense(
                        layers[-1],
                        params['hidden_layer'],
                        activation={'none': None,
                                    'relu': tf.nn.relu,
                                    'tanh': tf.nn.tanh,
                                    'sigmoid': \
                                        tf.nn.sigmoid}[params['activation']],
                        name='hidden_layer' + str(i+1)))
                if params['dropout']:
                    layers.append(
                        tf.layers.dropout(layers[-1],
                                          rate=params['dropout'],
                                          training=self.is_training))
            hidden_layer = layers[-1]
            output_layer = tf.layers.dense(
                hidden_layer,
                self.LABELS,
                activation=None,
                name='output_layer')
            self.predictions = tf.argmax(output_layer, axis=1, name='predictions')
            softmax = tf.nn.softmax(output_layer, axis=1, name='softmax')
            self.scores = softmax[:,1]

            # Training
            loss = tf.losses.sparse_softmax_cross_entropy(
                self.labels, output_layer, scope='loss')
            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer(
                params['learning_rate']).minimize(
                    loss, global_step=global_step, name='training')

            self.accuracy = tf.reduce_mean(
                tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
            self.recall = tf.reduce_sum(self.labels * self.predictions) / \
                          tf.reduce_sum(self.labels)
            self.precision = tf.reduce_sum(self.labels * self.predictions) / \
                          tf.reduce_sum(self.predictions)
            self.f1_score = 2 * self.recall * self.precision / \
                               (self.recall + self.precision)
            self.confusion_matrix = tf.reshape(
                tf.confusion_matrix(
                    self.labels, self.predictions, weights=tf.not_equal(
                        self.labels, self.predictions), dtype=tf.float32), [
                    1, self.LABELS, self.LABELS, 1])

            # Summaries
            summary_writer = tf.contrib.summary.create_file_writer(
                logdir, flush_millis=1 * 300)
            with summary_writer.as_default(), \
                   tf.contrib.summary.record_summaries_every_n_global_steps(10):
                self.summaries_train = [
                    tf.contrib.summary.scalar(
                        'train/loss',
                        loss),
                    tf.contrib.summary.scalar(
                        'train/accuracy',
                        self.accuracy),
                    tf.contrib.summary.scalar(
                        'train/f1_score',
                        self.f1_score)]
            with summary_writer.as_default(), \
                    tf.contrib.summary.always_record_summaries():
                self.summaries_test = [
                    tf.contrib.summary.scalar(
                        'test/accuracy',
                        self.accuracy),
                    tf.contrib.summary.scalar(
                        'test/f1_score',
                        self.f1_score),
                    tf.contrib.summary.image(
                        'test/confusion_matrix',
                        self.confusion_matrix)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(
                    session=self.session, graph=self.session.graph)

            # Saver
            tf.add_to_collection('end_points/is_training', self.is_training)
            tf.add_to_collection('end_points/array', self.array)
            tf.add_to_collection('end_points/labels', self.labels)
            tf.add_to_collection('end_points/scores', self.scores)
            tf.add_to_collection('end_points/training', self.training)
            tf.add_to_collection('end_points/loss', loss)
            self.saver = tf.train.Saver()

    def train(self, array, labels):
        self.session.run([self.training, self.summaries_train],
                         {self.array: array,
                          self.labels: labels,
                          self.is_training: True})

    def evaluate_summaries(self, array, labels):
        self.session.run(self.summaries_test,
                        {self.array: array,
                         self.labels: labels,
                         self.is_training: False})

    def evaluate_accuracy(self, array, labels):
        return self.session.run(self.accuracy,
                         {self.array: array,
                          self.labels: labels,
                          self.is_training: False})

    def evaluate_recall(self, array, labels):
        return self.session.run(self.recall,
                         {self.array: array,
                          self.labels: labels,
                          self.is_training: False})

    def evaluate_precision(self, array, labels):
        return self.session.run(self.precision,
                         {self.array: array,
                          self.labels: labels,
                          self.is_training: False})

    def evaluate_f1_score(self, array, labels):
        return self.session.run(self.f1_score,
                         {self.array: array,
                          self.labels: labels,
                          self.is_training: False})

    def save(self, path):
        return self.saver.save(self.session, path)

    def load_and_train(self, path, logdir):
        # Load the metagraph
        with self.session.graph.as_default():
            self.saver = tf.train.import_meta_graph(path + '.meta')

            # Attach the end points
            self.is_training = tf.get_collection('end_points/is_training')[0]
            self.array = tf.get_collection('end_points/array')[0]
            self.labels = tf.get_collection('end_points/labels')[0]
            self.scores = tf.get_collection('end_points/scores')[0]
            self.training = tf.get_collection('end_points/training')[0]
            loss = tf.get_collection('end_points/loss')[0]

        # Load the graph weights
        self.saver.restore(self.session, path)
        summary_writer = tf.contrib.summary.create_file_writer(
            logdir, flush_millis=1 * 300)
        with summary_writer.as_default(), \
               tf.contrib.summary.record_summaries_every_n_global_steps(10):
            self.summaries_train = [
                tf.contrib.summary.scalar(
                    'train/loss',
                    loss),
                tf.contrib.summary.scalar(
                    'train/accuracy',
                    self.accuracy),
                tf.contrib.summary.scalar(
                    'train/f1_score',
                    self.f1_score)]
        with summary_writer.as_default(), \
                tf.contrib.summary.always_record_summaries():
            self.summaries_test = [
                tf.contrib.summary.scalar(
                    'test/accuracy',
                    self.accuracy),
                tf.contrib.summary.scalar(
                    'test/f1_score',
                    self.f1_score),
                tf.contrib.summary.image(
                    'test/confusion_matrix',
                    self.confusion_matrix)]
        with summary_writer.as_default():
            tf.contrib.summary.initialize(
                session=self.session, graph=self.session.graph)



class NetworkPredict:
    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(
            graph=graph,
            config=tf.ConfigProto(
                inter_op_parallelism_threads=threads,
                intra_op_parallelism_threads=threads))

    def load(self, path):
        # Load the metagraph
        with self.session.graph.as_default():
            self.saver = tf.train.import_meta_graph(path + '.meta')

            # Attach the end points
            self.is_training = tf.get_collection('end_points/is_training')[0]
            self.array = tf.get_collection('end_points/array')[0]
            self.scores = tf.get_collection('end_points/scores')[0]

        # Load the graph weights
        self.saver.restore(self.session, path)

    def predict(self, array):
        return self.session.run(self.scores,
                                {self.array: array,
                                 self.is_training: False})
