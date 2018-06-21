# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path
import tensorflow as tf
class InferenceWrapperBase(object):
 def __init__(self):
  pass
 def build_model(self,model_config):
  tf.logging.fatal("Please implement build_model in subclass")
 def _create_restore_fn(self,checkpoint_path,saver):
  if tf.gfile.IsDirectory(checkpoint_path):
   checkpoint_path=tf.train.latest_checkpoint(checkpoint_path)
   if not checkpoint_path:
    raise ValueError("No checkpoint file found in: %s"%checkpoint_path)
  def _restore_fn(sess):
   tf.logging.info("Loading model from checkpoint: %s",checkpoint_path)
   saver.restore(sess,checkpoint_path)
   tf.logging.info("Successfully loaded checkpoint: %s",os.path.basename(checkpoint_path))
  return _restore_fn
 def build_graph_from_config(self,model_config,checkpoint_path):
  tf.logging.info("Building model.")
  self.build_model(model_config)
  saver=tf.train.Saver()
  return self._create_restore_fn(checkpoint_path,saver)
 def build_graph_from_proto(self,graph_def_file,saver_def_file,checkpoint_path):
  tf.logging.info("Loading GraphDef from file: %s",graph_def_file)
  graph_def=tf.GraphDef()
  with tf.gfile.FastGFile(graph_def_file,"rb")as f:
   graph_def.ParseFromString(f.read())
  tf.import_graph_def(graph_def,name="")
  tf.logging.info("Loading SaverDef from file: %s",saver_def_file)
  saver_def=tf.train.SaverDef()
  with tf.gfile.FastGFile(saver_def_file,"rb")as f:
   saver_def.ParseFromString(f.read())
  saver=tf.train.Saver(saver_def=saver_def)
  return self._create_restore_fn(checkpoint_path,saver)
 def feed_image(self,sess,encoded_image):
  tf.logging.fatal("Please implement feed_image in subclass")
 def inference_step(self,sess,input_feed,state_feed):
  tf.logging.fatal("Please implement inference_step in subclass")


