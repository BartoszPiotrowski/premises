#!/usr/bin/env python3
import numpy as np
import tensorflow as tf


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
            self.array_left = tf.placeholder(
                tf.float32, [None, self.NUM_OF_FEATURES], name='array')
            self.array_right = tf.placeholder(
                tf.float32, [None, self.NUM_OF_FEATURES], name='array')
            self.labels = tf.placeholder(tf.int64, [None], name='labels')
            self.is_training = tf.placeholder(tf.bool, [], name='is_training')

            # Computation
            layers_left = [self.array_left]
            layers_left.append(
                tf.layers.dense(
                    layers_left[-1],
                    params['hidden_layer'],
                    activation=tf.nn.relu,
                    name='layer_left_0'))
            if params['dropout']:
                layers_left.append(
                    tf.layers.dropout(layers_left[-1],
                                      rate=params['dropout'],
                                      training=self.is_training))

            layers_right = [self.array_right]
            layers_right.append(
                tf.layers.dense(
                    layers_right[-1],
                    params['hidden_layer'],
                    activation=tf.nn.relu,
                    name='layer_right_0'))
            if params['dropout']:
                layers_right.append(
                    tf.layers.dropout(layers_right[-1],
                                      rate=params['dropout'],
                                      training=self.is_training))

            layers_top = [tf.concat([layers_left[-1], layers_right[-1]], axis=1)]
            layers_top.append(
                tf.layers.dense(
                    layers_top[-1],
                    params['hidden_layer'],
                    activation=tf.nn.relu,
                    name='top_layer_0'))
            if params['dropout']:
                layers_top.append(
                    tf.layers.dropout(layers_right[-1],
                                      rate=params['dropout'],
                                      training=self.is_training))
            output_layer = tf.layers.dense(
                layers_top[-1],
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


            # Initialize variables
            self.session.run(tf.global_variables_initializer())

            # Saver
            tf.add_to_collection('end_points/array_left', self.array_left)
            tf.add_to_collection('end_points/array_right', self.array_right)
            tf.add_to_collection('end_points/is_training', self.is_training)
            tf.add_to_collection('end_points/labels', self.labels)
            tf.add_to_collection('end_points/scores', self.scores)
            tf.add_to_collection('end_points/training', self.training)
            tf.add_to_collection('end_points/accuracy', self.accuracy)
            tf.add_to_collection('end_points/recall', self.recall)
            tf.add_to_collection('end_points/precision', self.precision)
            tf.add_to_collection('end_points/f1_score', self.f1_score)
            self.saver = tf.train.Saver()

    def train(self, array, labels):
        self.session.run(self.training,
                         {self.array_left: array[0],
                          self.array_right: array[1],
                          self.labels: labels,
                          self.is_training: True})

    def evaluate_accuracy(self, array, labels):
        return self.session.run(self.accuracy,
                         {self.array_left: array[0],
                          self.array_right: array[1],
                          self.labels: labels,
                          self.is_training: False})

    def evaluate_recall(self, array, labels):
        return self.session.run(self.recall,
                         {self.array_left: array[0],
                          self.array_right: array[1],
                          self.labels: labels,
                          self.is_training: False})

    def evaluate_precision(self, array, labels):
        return self.session.run(self.precision,
                         {self.array_left: array[0],
                          self.array_right: array[1],
                          self.labels: labels,
                          self.is_training: False})

    def evaluate_f1_score(self, array, labels):
        return self.session.run(self.f1_score,
                         {self.array_left: array[0],
                          self.array_right: array[1],
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
            self.array_left = tf.get_collection('end_points/array_left')[0]
            self.array_right = tf.get_collection('end_points/array_right')[0]
            self.labels = tf.get_collection('end_points/labels')[0]
            self.scores = tf.get_collection('end_points/scores')[0]
            self.training = tf.get_collection('end_points/training')[0]
            self.accuracy = tf.get_collection('end_points/accuracy')[0]
            self.recall = tf.get_collection('end_points/recall')[0]
            self.precision = tf.get_collection('end_points/precision')[0]
            self.f1_score = tf.get_collection('end_points/f1_score')[0]

        # Load the graph weights
        self.saver.restore(self.session, path)


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
            self.array_left = tf.get_collection('end_points/array_left')[0]
            self.array_right = tf.get_collection('end_points/array_right')[0]
            self.scores = tf.get_collection('end_points/scores')[0]

        # Load the graph weights
        self.saver.restore(self.session, path)

    def predict(self, array):
        return self.session.run(self.scores,
                                {self.array_left: array[0],
                                 self.array_right: array[1],
                                 self.is_training: False})
