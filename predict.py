import sys
# sys.path.append('/usr/local/lib/python3.5/dist-packages/')
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import http.client, urllib.request, urllib.parse, urllib.error, base64
import json
import warnings
warnings.filterwarnings("ignore")
import pickle
import sentencepiece as spm



media_app='whatsapp'# modify your media app here

module_url = "https://tfhub.dev/google/universal-sentence-encoder-lite/2"
embed = hub.Module(module_url)
tf.logging.set_verbosity(tf.logging.WARN)

module = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-lite/2")
input_placeholder = tf.sparse_placeholder(tf.int64, shape=[None, None])
encodings = module(
    inputs=dict(
        values=input_placeholder.values,
        indices=input_placeholder.indices,
        dense_shape=input_placeholder.dense_shape))

with tf.Session() as sess:
    spm_path = sess.run(module(signature="spm_path"))

sp = spm.SentencePieceProcessor()
sp.Load(spm_path)
print("SentencePiece model loaded at {}.".format(spm_path))

def process_to_IDs_in_sparse_format(sp, sentences):
  # An utility method that processes sentences with the sentence piece processor
  # 'sp' and returns the results in tf.SparseTensor-similar format:
  # (values, indices, dense_shape)
    ids = [sp.EncodeAsIds(x) for x in sentences]
    max_len = max(len(x) for x in ids)
    dense_shape=(len(ids), max_len)
    values=[item for sublist in ids for item in sublist]
    indices=[[row,col] for row in range(len(ids)) for col in range(len(ids[row]))]
    return (values, indices, dense_shape)

def embed_sentence_lite(sentences):
    messages = sentences
    values, indices, dense_shape = process_to_IDs_in_sparse_format(sp, messages)

    # Reduce logging output.
    tf.logging.set_verbosity(tf.logging.ERROR)

    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        message_embeddings = session.run(
          encodings,
          feed_dict={input_placeholder.values: values,
                    input_placeholder.indices: indices,
                    input_placeholder.dense_shape: dense_shape})
    
    return message_embeddings


def find_closest(sentence_rep,query_rep,K):
    top_K = np.argsort(np.sqrt((np.sum(np.square(sentence_rep - query_rep),axis=1))))[:K]
    return top_K

f = open('res/'+media_app+'/other_embeddings.p','rb')
other_embeddings = pickle.load(f)
f.close()

f = open('res/'+media_app+'/your_embeddings.p','rb')
your_embeddings = pickle.load(f)
f.close()

f = open('res/'+media_app+'/dilogues.p','rb')
pr_to_sp = pickle.load(f)
f.close()


f = open('res/'+media_app+'/your_sents.p','rb')
your_sentences = pickle.load(f)
f.close()

keys = list(pr_to_sp.keys())

f = open('res/'+media_app+'/key_embeddings.p','rb')
key_embeddings = pickle.load(f)
f.close()


def speak_like_me(query,K,your_embeddings,other_embeddings,your_sen):
    other_query = [query]
    query_embedding = embed_sentence_lite(other_query)
    closest_your = find_closest(your_embeddings,query_embedding,K)
    for cl in closest_your:
        print(your_sentences[cl])


def respond_like_me(query,K,key_embeddings,keys):
    other_query = [query]
    query_embedding = embed_sentence_lite(other_query)
    closest_other = find_closest(key_embeddings,query_embedding,K+2)
    for k in closest_other[3:]:
        print(pr_to_sp[keys[k]])



respond_like_me("bye",15,key_embeddings,keys)



speak_like_me("bye",10,your_embeddings,other_embeddings,your_sentences)