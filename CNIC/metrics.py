from __future__ import absolute_import
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

FLAGS=tf.flags.FLAGS
tf.flags.DEFINE_string("checkpoint_path","","Model checkpoint file or directory containing a " "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file","","Text file containing the vocabulary.")
tf.flags.DEFINE_string("image_path","","Directory of image files.")
tf.flags.DEFINE_string("temp_path","/tmp","Directory of temporary files.")
tf.flags.DEFINE_string("test_json_path","","Path of test json path.")
tf.flags.DEFINE_string("keyword_pickle_file","/tmp/features.pkl","Attibute pickle file.")
tf.logging.set_verbosity(tf.logging.INFO)
def main(_):
 with open(FLAGS.keyword_pickle_file,'r')as f:
  keyword_data=cPickle.load(f)
 with open(FLAGS.test_json_path)as f:
  test_json=json.load(f)
 id_to_filename=test_json['images']
 id_to_path=[{'path':os.path.join(FLAGS.image_path,x['file_name']),'id':x['id']}for x in id_to_filename]
 result_json=[]
 g=tf.Graph()
 with g.as_default():
  model=inference_wrapper.InferenceWrapper()
  restore_fn=model.build_graph_from_config(configuration.ModelConfig(),FLAGS.checkpoint_path)
 g.finalize()
 vocab=vocabulary.Vocabulary(FLAGS.vocab_file)
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
 with open(os.path.join(FLAGS.temp_path,"result.json"),'w')as f:
  json.dump(result_json,f)
 coco=COCO(FLAGS.test_json_path)
 cocoRes=coco.loadRes(os.path.join(FLAGS.temp_path,"result.json"))
 cocoEval=COCOEvalCap(coco,cocoRes)
 cocoEval.evaluate()
if __name__=="__main__":
 tf.app.run()

