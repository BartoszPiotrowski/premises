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

    def construct(self, params):
        self.NUM_OF_FEATURES = params['num_of_features']
        self.LABELS = 2
        with self.session.graph.as_default():
            # Inputs
            self.array = tf.placeholder(
                tf.float32, [None, self.NUM_OF_FEATURES], name="proofs")
            self.labels = tf.placeholder(tf.int64, [None], name="labels")

            # Computation
            flattened_array = tf.layers.flatten(self.array, name="flatten")
            layers = [flattened_array]
            for i in range(params["layers"]):
                layers.append(
                    tf.layers.dense(
                        layers[i],
                        params["hidden_layer"],
                        activation={'none': None,
                                    'relu': tf.nn.relu,
                                    'tanh': tf.nn.tanh,
                                    'sigmoid': tf.nn.sigmoid}[params["activation"]],
                        name="hidden_layer" + str(i+1)))
            hidden_layer = layers[-1]
            output_layer = tf.layers.dense(
                hidden_layer,
                self.LABELS,
                activation=None,
                name="output_layer")
            self.predictions = tf.argmax(output_layer, axis=1)

            # Training
            loss = tf.losses.sparse_softmax_cross_entropy(
                self.labels, output_layer, scope="loss")
            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer(0.03).minimize(
                loss, global_step=global_step, name="training")

            # Summaries
            accuracy = tf.reduce_mean(
                tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
            self.accuracy = tf.reduce_mean(
                tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
            confusion_matrix = tf.reshape(
                tf.confusion_matrix(
                    self.labels, self.predictions, weights=tf.not_equal(
                        self.labels, self.predictions), dtype=tf.float32), [
                    1, self.LABELS, self.LABELS, 1])

            summary_writer = tf.contrib.summary.create_file_writer(
                params["logdir"], flush_millis=1 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), \
                    tf.contrib.summary.record_summaries_every_n_global_steps(100):
                self.summaries["train"] = [
                    tf.contrib.summary.scalar(
                        "train/loss",
                        loss),
                    tf.contrib.summary.scalar(
                        "train/accuracy",
                        accuracy)]
            with summary_writer.as_default(), \
                    tf.contrib.summary.always_record_summaries():
                self.summaries['test'] = [
                    tf.contrib.summary.scalar(
                        "test/accuracy",
                        accuracy),
                    tf.contrib.summary.image(
                        "test/confusion_matrix",
                        confusion_matrix)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(
                    session=self.session, graph=self.session.graph)

            # Saver
            self.saver = tf.train.Saver()

    def train(self, array, labels):
        self.session.run([self.training, self.summaries["train"]],
                         {self.array: array, self.labels: labels})

    def evaluate(self, dataset, array, labels):
        self.session.run(self.summaries[dataset],
                        {self.array: array, self.labels: labels})

    def evaluate_accuracy(self, dataset, array, labels):
        return self.session.run(self.accuracy,
                        {self.array: array, self.labels: labels})

    def predict(self, array, labels):
        return self.session.run(self.predictions,
                        {self.array: array, self.labels: labels})

    def save(self, path):
        self.saver.save(self.session, path)
