#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import os
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import tensorflow as tf
import configuration
import inference_wrapper
from inference_utils import caption_generator
from inference_utils import vocabulary
import json
import cPickle
import numpy as np
import six

def main(test_json_path):
 model_list = []
 for i in xrange(4):
  model_list.append(os.path.join(os.path.dirname(__file__),'models/train','model.ckpt'+str(i)))
 var_list = tf.contrib.framework.list_variables(model_list[0])
 var_values, var_dtypes = {}, {}
 for (name, shape) in var_list:
  if not name.startswith("global_step"):
   var_values[name] = np.zeros(shape)
 for model_path in model_list:
  reader = tf.contrib.framework.load_checkpoint(model_path)
  for name in var_values:
   tensor = reader.get_tensor(name)
   var_dtypes[name] = tensor.dtype
   var_values[name] += tensor
 for name in var_values:
  var_values[name] /= len(model_list)
 tf_vars = [tf.get_variable(v, shape=var_values[v].shape, dtype=var_dtypes[name]) for v in var_values]
 placeholders = [tf.placeholder(v.dtype, shape=v.get_shape()) for v in tf_vars]
 assign_ops = [tf.assign(v, p) for (v, p) in zip(tf_vars, placeholders)]
 global_step = tf.Variable(0, name="global_step", trainable=False, dtype=tf.int32)
 saver = tf.train.Saver(tf.all_variables())
 with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for p,assign_op,(name,value) in zip(placeholders,assign_ops,six.iteritems(var_values)):
   sess.run(assign_op, {p: value})
  saver.save(sess, os.path.join(os.path.dirname(__file__),'models/tmp/model.ckpt'), global_step=global_step)
 with open(os.path.join(os.path.dirname(__file__),'data/features.pkl'),'r')as f:
  keyword_data=cPickle.load(f)
 with open(test_json_path)as f:
  test_json=json.load(f)
 id_to_filename=test_json['images']
 id_to_path=[{'path':os.path.join('./Data/test',x['file_name']),'id':x['id']}for x in id_to_filename]
 result_json=[]
 g=tf.Graph()
 with g.as_default():
  model=inference_wrapper.InferenceWrapper()
  restore_fn=model.build_graph_from_config(configuration.ModelConfig(),os.path.join(os.path.dirname(__file__),'models/tmp/model.ckpt-0'))
 g.finalize()
 vocab=vocabulary.Vocabulary(os.path.join('./Data/word_counts.txt'))
 with tf.Session(graph=g)as sess:
  restore_fn(sess)
  generator=caption_generator.CaptionGenerator(model,vocab)
  for data in id_to_path:
   filename=data['path']
   with tf.gfile.GFile(filename,"r")as f:
    image=f.read()
   captions=generator.beam_search(sess,image,keyword_data[os.path.basename(filename)])
   print("Captions for image %s:"%os.path.basename(filename))
   result={'image_id':data['id'],'caption':(" ".join([vocab.id_to_word(w)for w in captions[0].sentence[1:-1]])).decode('utf-8')}
   print(result)
   result_json.append(result)
 with open(os.path.join(os.path.dirname(__file__),"result.json"),'w')as f:
  json.dump(result_json,f)
 coco=COCO(test_json_path)
 cocoRes=coco.loadRes(os.path.join(os.path.dirname(__file__),"result.json"))
 cocoEval=COCOEvalCap(coco,cocoRes)
 cocoEval.evaluate()


