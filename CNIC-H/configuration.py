# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
class ModelConfig(object):
 def __init__(self):
  self.input_file_pattern=None
  self.image_format="jpeg"
  self.values_per_input_shard=4278
  self.input_queue_capacity_factor=2
  self.num_input_reader_threads=1
  self.image_feature_name="image/data"
  self.caption_feature_name="image/caption_ids"
  self.word_prob_feature_name="image/word_prob"
  self.vocab_size=12000
  self.num_preprocess_threads=4
  self.batch_size=32
  self.inception_checkpoint_file=None
  self.image_height=299
  self.image_width=299
  self.initializer_scale=0.08
  self.embedding_size=512
  self.num_lstm_units=512
  self.lstm_dropout_keep_prob=0.7
class TrainingConfig(object):
 def __init__(self):
  self.num_examples_per_epoch=1095000
  self.optimizer="SGD"
  self.initial_learning_rate=2.0
  self.learning_rate_decay_factor=0.5
  self.num_epochs_per_decay=8.0
  self.train_inception_learning_rate=0.0005
  self.clip_gradients=5.0
  self.max_checkpoints_to_keep=500


