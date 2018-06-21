# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
class Vocabulary(object):
 def __init__(self,vocab_file,start_word="<S>",end_word="</S>",unk_word="<UNK>"):
  if not tf.gfile.Exists(vocab_file):
   tf.logging.fatal("Vocab file %s not found.",vocab_file)
  tf.logging.info("Initializing vocabulary from file: %s",vocab_file)
  with tf.gfile.GFile(vocab_file,mode="r")as f:
   reverse_vocab=list(f.readlines())
  reverse_vocab=[line.split()[0]for line in reverse_vocab]
  assert start_word in reverse_vocab
  assert end_word in reverse_vocab
  if unk_word not in reverse_vocab:
   reverse_vocab.append(unk_word)
  vocab=dict([(x,y)for(y,x)in enumerate(reverse_vocab)])
  tf.logging.info("Created vocabulary with %d words"%len(vocab))
  self.vocab=vocab 
  self.reverse_vocab=reverse_vocab 
  self.start_id=vocab[start_word]
  self.end_id=vocab[end_word]
  self.unk_id=vocab[unk_word]
 def word_to_id(self,word):
  if word in self.vocab:
   return self.vocab[word]
  else:
   return self.unk_id
 def id_to_word(self,word_id):
  if word_id>=len(self.reverse_vocab):
   return self.reverse_vocab[self.unk_id]
  else:
   return self.reverse_vocab[word_id]


