# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import numpy as np
from detector import Detector
from util import load_image
import matplotlib.pyplot as plt
import cPickle
import os
import math
import configuration
import inference_wrapper
from inference_utils import caption_generator
from inference_utils import vocabulary
import json
import shutil
from bottle import run,post,request,response
import time
import ntpath
start_time=time.time()
model_path='./models/model-2'
wordset_path='./models/words.pkl'
weight_path='./models/caffe_layers_value.pickle'
drop_words=[u'的',u'着',u'有',u'在',u'上',u'里',u'和',u'旁',u'前',u'、',u'后面',u'前站',u'下',u'中']
with open(wordset_path)as f:
 wordset=cPickle.load(f)
inv_map={}
for i,x in enumerate(wordset):
 inv_map[i]=x
n_labels=len(wordset)
images_tf=tf.placeholder(tf.float32,[None,224,224,3],name="images")
labels_tf=tf.placeholder(tf.int64,[None],name='labels')
detector=Detector(weight_path,n_labels)
c1,c2,c3,c4,conv5,conv6,gap,output=detector.inference(images_tf)
sig_output=tf.nn.sigmoid(output)
classmap=detector.get_classmap(labels_tf,conv6)
sess1=tf.InteractiveSession()
saver=tf.train.Saver()
saver.restore(sess1,model_path)
g=tf.Graph()
with g.as_default():
 model=inference_wrapper.InferenceWrapper()
 restore_fn=model.build_graph_from_config(configuration.ModelConfig(),'./models/model.ckpt-633626')
g.finalize()
vocab=vocabulary.Vocabulary('./models/word_counts.txt')
sess2=tf.InteractiveSession(graph=g)
restore_fn(sess2)
generator=caption_generator.CaptionGenerator(model,vocab)
print time.time()-start_time
@post('/process')
def my_process():
 start_time=time.time()
 req_obj=json.loads(request.body.read())
 print 'extract feature from %s'%req_obj['input']
 image=load_image(req_obj['input'])
 conv6_val,output_val,sig_output_val=sess1.run([conv6,output,sig_output],feed_dict={images_tf:[image]})
 rlabel_predictions=sig_output_val.round()
 mask=np.ones((1,n_labels))
 last= rlabel_predictions==mask
 xx=np.argsort(sig_output_val[0])[::-1]
 prob={}
 shutil.copyfile(req_obj['input'],os.path.join('/tmp/demo/freeze',ntpath.basename(req_obj['input'])))
 shutil.rmtree(req_obj['output'])
 os.makedirs(req_obj['output'])
 shutil.copyfile(os.path.join('/tmp/demo/freeze',ntpath.basename(req_obj['input'])),os.path.join(req_obj['output'],ntpath.basename(req_obj['input'])))
 print time.time()-start_time
 start_time=time.time()
 for x in xx:
  if inv_map[x]not in drop_words and sig_output_val[0][x]>req_obj['threshold']:
   start_time=time.time()
   classmap_vals=sess1.run(classmap,feed_dict={labels_tf:[x],conv6:conv6_val})
   print time.time()-start_time
   start_time=time.time()
   classmap_vis=map(lambda x:((x-x.min())/(x.max()-x.min())),classmap_vals)
   for vis,ori in zip(classmap_vis,[image]):
    print inv_map[x],sig_output_val[0][x]
    prob[inv_map[x]]=float(sig_output_val[0][x])
    fig=plt.gcf()
    plt.axis('off')
    plt.imshow(ori)
    plt.imshow(vis,cmap=plt.cm.jet,alpha=0.5,interpolation='nearest')
    fig.savefig(os.path.join(req_obj['output'],inv_map[x]+'.png'),dpi=100)
   print 'sace:',time.time()-start_time
 feature=sig_output_val[0]
 print time.time()-start_time
 start_time=time.time()
 with tf.gfile.GFile(req_obj['input'],"r")as f:
  image=f.read()
 captions=generator.beam_search(sess2,image,feature)
 print time.time()-start_time
 result={'filename':req_obj['input'],'caption':(" ".join([vocab.id_to_word(w)for w in captions[0].sentence[1:-1]])).decode('utf-8'),'prob':prob}
 print result
 return json.dumps(result)
run(host='localhost',port=8080,debug=True)


