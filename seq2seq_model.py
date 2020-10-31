# Vellore Institute of Technology, Vellore
#TITLE: IMPLEMETING A CHATBOT USING RNN AND SEQ2SEQ MODELING
#MADE BY: SWAMI VENKAT (16BCE2270)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

#from tensorflow.models.rnn.translate import data_utils
#ModuleNotFoundError: No module named 'tensorflow.models'

import data_utils

class Seq2SeqModel(object):
 
  def __init__(self, source_vocab_size, target_vocab_size, buckets, size,
               num_layers, max_gradient_norm, batch_size, learning_rate,
               learning_rate_decay_factor, use_lstm=False,
               num_samples=512, forward_only=False):
    
    self.source_vocab_size = source_vocab_size
    self.target_vocab_size = target_vocab_size
    self.buckets = buckets
    self.batch_size = batch_size
    self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
    self.learning_rate_decay_op = self.learning_rate.assign(
        self.learning_rate * learning_rate_decay_factor)
    self.global_step = tf.Variable(0, trainable=False)

    # If we use sampled softmax, we need an output projection.
    output_projection = None
    softmax_loss_function = None
    # Sampled softmax only makes sense if we sample less than vocabulary size.
    if num_samples > 0 and num_samples < self.target_vocab_size:
      w = tf.get_variable("proj_w", [size, self.target_vocab_size])
      w_t = tf.transpose(w)
      b = tf.get_variable("proj_b", [self.target_vocab_size])
      output_projection = (w, b)

      def sampled_loss(inputs, labels):
        labels = tf.reshape(labels, [-1, 1])
        return tf.nn.sampled_softmax_loss(w_t, b, inputs, labels, num_samples,
                self.target_vocab_size)
      softmax_loss_function = sampled_loss

    # Create the internal multi-layer cell for our RNN.
    single_cell = tf.nn.rnn_cell.GRUCell(size)
    if use_lstm:
      single_cell = tf.nn.rnn_cell.BasicLSTMCell(size)
    cell = single_cell
    if num_layers > 1:
      cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)

    # The seq2seq function: we use embedding for the input and attention.
    def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
      return tf.nn.seq2seq.embedding_attention_seq2seq(
          encoder_inputs, decoder_inputs, cell,
          num_encoder_symbols=source_vocab_size,
          num_decoder_symbols=target_vocab_size,
          embedding_size=size,
          output_projection=output_projection,
          feed_previous=do_decode)

    # Feeds for inputs.
    self.encoder_inputs = []
    self.decoder_inputs = []
    self.target_weights = []
    for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
      self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="encoder{0}".format(i)))
    for i in xrange(buckets[-1][1] + 1):
      self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="decoder{0}".format(i)))
      self.target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                name="weight{0}".format(i)))
