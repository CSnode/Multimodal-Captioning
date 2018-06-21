from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from ops import image_embedding
from ops import image_processing
from ops import inputs as input_ops

class ShowAndTellModel(object):
 def __init__(self,config,mode,train_inception=False):
  assert mode in["train","eval","inference"]
  self.config=config
  self.mode=mode
  self.train_inception=train_inception
  self.reader=tf.TFRecordReader()
  self.initializer=tf.random_uniform_initializer(minval=-self.config.initializer_scale,maxval=self.config.initializer_scale)
  self.images=None
  self.word_prob=None
  self.input_seqs=None
  self.target_seqs=None
  self.input_mask=None
  self.image_embeddings=None
  self.word_prob_embeddings=None
  self.lstm_zero_state=None
  self.seq_embeddings=None
  self.total_loss=None
  self.target_cross_entropy_losses=None
  self.target_cross_entropy_loss_weights=None
  self.inception_variables=[]
  self.init_fn=None
  self.global_step=None
 def is_training(self):
  return self.mode=="train"
 def process_image(self,encoded_image,thread_id=0):
  return image_processing.process_image(encoded_image,is_training=self.is_training(),height=self.config.image_height,width=self.config.image_width,thread_id=thread_id,image_format=self.config.image_format)
 def build_inputs(self):
  if self.mode=="inference":
   image_feed=tf.placeholder(dtype=tf.string,shape=[],name="image_feed")
   word_prob_feed=tf.placeholder(dtype=tf.float32,shape=[319],name="word_prob_feed")
   input_feed=tf.placeholder(dtype=tf.int64,shape=[None],name="input_feed")
   images=tf.expand_dims(self.process_image(image_feed),0)
   word_prob=tf.expand_dims(word_prob_feed,0)
   input_seqs=tf.expand_dims(input_feed,1)
   target_seqs=None
   input_mask=None
  else:
   input_queue=input_ops.prefetch_input_data(self.reader,self.config.input_file_pattern,is_training=self.is_training(),batch_size=self.config.batch_size,values_per_shard=self.config.values_per_input_shard,input_queue_capacity_factor=self.config.input_queue_capacity_factor,num_reader_threads=self.config.num_input_reader_threads)
   assert self.config.num_preprocess_threads%2==0
   images_word_prob_and_captions=[]
   for thread_id in range(self.config.num_preprocess_threads):
    serialized_sequence_example=input_queue.dequeue()
    encoded_image,caption,word_prob=input_ops.parse_sequence_example(serialized_sequence_example,image_feature=self.config.image_feature_name,word_prob_feature=self.config.word_prob_feature_name,caption_feature=self.config.caption_feature_name)
    image=self.process_image(encoded_image,thread_id=thread_id)
    images_word_prob_and_captions.append([image,word_prob,caption])
   queue_capacity=(2*self.config.num_preprocess_threads*self.config.batch_size)
   images,word_prob,input_seqs,target_seqs,input_mask=(input_ops.batch_with_dynamic_pad(images_word_prob_and_captions,batch_size=self.config.batch_size,queue_capacity=queue_capacity))
  self.images=images
  self.word_prob=word_prob
  self.input_seqs=input_seqs
  self.target_seqs=target_seqs
  self.input_mask=input_mask
 def build_image_embeddings(self):
  inception_output=image_embedding.inception_v3(self.images,trainable=self.train_inception,is_training=self.is_training())
  self.inception_variables=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="InceptionV3")
  with tf.variable_scope("image_embedding")as scope:
   image_embeddings=tf.contrib.layers.fully_connected(inputs=inception_output,num_outputs=self.config.embedding_size,activation_fn=None,weights_initializer=self.initializer,biases_initializer=None,scope=scope)
  tf.constant(self.config.embedding_size,name="embedding_size")
  self.image_embeddings=image_embeddings
 def build_word_prob_embeddings(self):
  with tf.variable_scope("word_prob_embedding")as scope:
   word_prob_embeddings=tf.contrib.layers.fully_connected(inputs=self.word_prob,num_outputs=self.config.embedding_size,activation_fn=None,weights_initializer=self.initializer,biases_initializer=None,scope=scope)
  self.word_prob_embeddings=word_prob_embeddings
 def build_lstm_zero_state(self):
  with tf.variable_scope("lstm_zero_state")as scope:
   cell_state=tf.zeros([self.word_prob_embeddings.get_shape()[0],self.word_prob_embeddings.get_shape()[1]],tf.float32)
   hidden_state=self.word_prob_embeddings
   lstm_zero_state=tf.contrib.rnn.LSTMStateTuple(cell_state,hidden_state)
  self.lstm_zero_state=lstm_zero_state
 def build_seq_embeddings(self):
  with tf.variable_scope("seq_embedding"),tf.device("/cpu:0"):
   embedding_map=tf.get_variable(name="map",shape=[self.config.vocab_size,self.config.embedding_size],initializer=self.initializer)
   seq_embeddings=tf.nn.embedding_lookup(embedding_map,self.input_seqs)
  self.seq_embeddings=seq_embeddings
 def build_model(self):
  lstm_cell=tf.contrib.rnn.BasicLSTMCell(num_units=self.config.num_lstm_units,state_is_tuple=True)
  if self.mode=="train":
   lstm_cell=tf.contrib.rnn.DropoutWrapper(lstm_cell,input_keep_prob=self.config.lstm_dropout_keep_prob,output_keep_prob=self.config.lstm_dropout_keep_prob)
  with tf.variable_scope("lstm",initializer=self.initializer)as lstm_scope:
   zero_state=self.lstm_zero_state
   _,initial_state=lstm_cell(self.image_embeddings,zero_state)
   lstm_scope.reuse_variables()
   if self.mode=="inference":
    tf.concat(axis=1,values=initial_state,name="initial_state")
    state_feed=tf.placeholder(dtype=tf.float32,shape=[None,sum(lstm_cell.state_size)],name="state_feed")
    state_tuple=tf.split(value=state_feed,num_or_size_splits=2,axis=1)
    lstm_outputs,state_tuple=lstm_cell(inputs=tf.squeeze(self.seq_embeddings,axis=[1]),state=state_tuple)
    tf.concat(axis=1,values=state_tuple,name="state")
   else:
    sequence_length=tf.reduce_sum(self.input_mask,1)
    lstm_outputs,_=tf.nn.dynamic_rnn(cell=lstm_cell,inputs=self.seq_embeddings,sequence_length=sequence_length,initial_state=initial_state,dtype=tf.float32,scope=lstm_scope)
  lstm_outputs=tf.reshape(lstm_outputs,[-1,lstm_cell.output_size])
  with tf.variable_scope("logits")as logits_scope:
   logits=tf.contrib.layers.fully_connected(inputs=lstm_outputs,num_outputs=self.config.vocab_size,activation_fn=None,weights_initializer=self.initializer,scope=logits_scope)
  if self.mode=="inference":
   tf.nn.softmax(logits,name="softmax")
  else:
   targets=tf.reshape(self.target_seqs,[-1])
   weights=tf.to_float(tf.reshape(self.input_mask,[-1]))
   losses=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets,logits=logits)
   batch_loss=tf.div(tf.reduce_sum(tf.multiply(losses,weights)),tf.reduce_sum(weights),name="batch_loss")
   tf.losses.add_loss(batch_loss)
   total_loss=tf.losses.get_total_loss()
   tf.summary.scalar("losses/batch_loss",batch_loss)
   tf.summary.scalar("losses/total_loss",total_loss)
   for var in tf.trainable_variables():
    tf.summary.histogram("parameters/"+var.op.name,var)
   self.total_loss=total_loss
   self.target_cross_entropy_losses=losses 
   self.target_cross_entropy_loss_weights=weights 
 def setup_inception_initializer(self):
  if self.mode!="inference":
   saver=tf.train.Saver(self.inception_variables)
   def restore_fn(sess):
    tf.logging.info("Restoring Inception variables from checkpoint file %s",self.config.inception_checkpoint_file)
    saver.restore(sess,self.config.inception_checkpoint_file)
   self.init_fn=restore_fn
 def setup_global_step(self):
  global_step=tf.Variable(initial_value=0,name="global_step",trainable=False,collections=[tf.GraphKeys.GLOBAL_STEP,tf.GraphKeys.GLOBAL_VARIABLES])
  self.global_step=global_step
 def build(self):
  self.build_inputs()
  self.build_image_embeddings()
  self.build_word_prob_embeddings()
  self.build_lstm_zero_state()
  self.build_seq_embeddings()
  self.build_model()
  self.setup_inception_initializer()
  self.setup_global_step()