from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import os
import os.path
import time
import re
import numpy as np
import tensorflow as tf
import configuration
import show_and_tell_model

FLAGS=tf.flags.FLAGS
tf.flags.DEFINE_string("input_file_pattern","","File pattern of sharded TFRecord input files.")
tf.flags.DEFINE_string("checkpoint_dir","","Directory containing model checkpoints.")
tf.flags.DEFINE_string("eval_dir","","Directory to write event logs.")
tf.flags.DEFINE_integer("eval_interval_secs",300,"Interval between evaluation runs.")
tf.flags.DEFINE_integer("num_eval_examples",5000,"Number of examples for evaluation.")
tf.flags.DEFINE_integer("min_global_step",1000,"Minimum global step to run evaluation.")
tf.logging.set_verbosity(tf.logging.INFO)

def evaluate_model(sess,model,global_step,summary_writer,summary_op):
 summary_str=sess.run(summary_op)
 summary_writer.add_summary(summary_str,global_step)
 num_eval_batches=int(math.ceil(FLAGS.num_eval_examples/model.config.batch_size))
 start_time=time.time()
 sum_losses=0.
 sum_weights=0.
 for i in xrange(num_eval_batches):
  cross_entropy_losses,weights=sess.run([model.target_cross_entropy_losses,model.target_cross_entropy_loss_weights])
  sum_losses+=np.sum(cross_entropy_losses*weights)
  sum_weights+=np.sum(weights)
  if not i%100:
   tf.logging.info("Computed losses for %d of %d batches.",i+1,num_eval_batches)
 eval_time=time.time()-start_time
 perplexity=math.exp(sum_losses/sum_weights)
 tf.logging.info("Perplexity = %f (%.2g sec)",perplexity,eval_time)
 summary=tf.Summary()
 value=summary.value.add()
 value.simple_value=perplexity
 value.tag="Perplexity"
 summary_writer.add_summary(summary,global_step)
 summary_writer.flush()
 tf.logging.info("Finished processing evaluation at global step %d.",global_step)

def run_once(model,saver,model_path,summary_writer,summary_op):
 if not model_path:
  tf.logging.info("Skipping evaluation. No checkpoint found in: %s",FLAGS.checkpoint_dir)
  return
 with tf.Session()as sess:
  tf.logging.info("Loading model from checkpoint: %s",model_path)
  saver.restore(sess,model_path)
  global_step=tf.train.global_step(sess,model.global_step.name)
  tf.logging.info("Successfully loaded %s at global step = %d.",os.path.basename(model_path),global_step)
  if global_step<FLAGS.min_global_step:
   tf.logging.info("Skipping evaluation. Global step = %d < %d",global_step,FLAGS.min_global_step)
   return
  coord=tf.train.Coordinator()
  threads=tf.train.start_queue_runners(coord=coord)
  try:
   evaluate_model(sess=sess,model=model,global_step=global_step,summary_writer=summary_writer,summary_op=summary_op)
  except Exception,e: 
   tf.logging.error("Evaluation failed.")
   coord.request_stop(e)
  coord.request_stop()
  coord.join(threads,stop_grace_period_secs=10)
def run():
 eval_dir=FLAGS.eval_dir
 if not tf.gfile.IsDirectory(eval_dir):
  tf.logging.info("Creating eval directory: %s",eval_dir)
  tf.gfile.MakeDirs(eval_dir)
 g=tf.Graph()
 with g.as_default():
  model_config=configuration.ModelConfig()
  model_config.input_file_pattern=FLAGS.input_file_pattern
  model=show_and_tell_model.ShowAndTellModel(model_config,mode="eval")
  model.build()
  saver=tf.train.Saver()
  summary_op=tf.summary.merge_all()
  summary_writer=tf.summary.FileWriter(eval_dir)
  g.finalize()
  i=-1
  while True:
   start=time.time()
   tf.logging.info("Starting evaluation at "+time.strftime("%Y-%m-%d-%H:%M:%S",time.localtime()))
   current_filenames=os.listdir(FLAGS.checkpoint_dir)
   nums=[]
   for x in current_filenames:
    if x[-5:]=='index':
     nums.append(int(re.findall(r'\d+',x)[0]))
   nums.sort()
   for x in nums:
    if x>i:
     run_once(model,saver,os.path.join(FLAGS.checkpoint_dir,'model.ckpt-'+str(x)),summary_writer,summary_op)
     i=x
     break
   time_to_next_eval=start+FLAGS.eval_interval_secs-time.time()
   if time_to_next_eval>0:
    time.sleep(time_to_next_eval)

def main(unused_argv):
 assert FLAGS.input_file_pattern,"--input_file_pattern is required"
 assert FLAGS.checkpoint_dir,"--checkpoint_dir is required"
 assert FLAGS.eval_dir,"--eval_dir is required"
 run()
if __name__=="__main__":
 tf.app.run()

