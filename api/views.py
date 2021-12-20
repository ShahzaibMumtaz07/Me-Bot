from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.views import APIView
from django.conf import settings
from decouple import config

from api.serializers import BotSerializer


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



class Bot(APIView):
    media_app='whatsapp'# modify your media app here
    sp = spm.SentencePieceProcessor()
    sp.Load(settings.SPM_PATH)
    

    def process_to_IDs_in_sparse_format(self,sp, sentences):
    # An utility method that processes sentences with the sentence piece processor
    # 'sp' and returns the results in tf.SparseTensor-similar format:
    # (values, indices, dense_shape)
        ids = [sp.EncodeAsIds(x) for x in sentences]
        max_len = max(len(x) for x in ids)
        dense_shape=(len(ids), max_len)
        values=[item for sublist in ids for item in sublist]
        indices=[[row,col] for row in range(len(ids)) for col in range(len(ids[row]))]
        return (values, indices, dense_shape)

    def embed_sentence_lite(self, sentences):
        messages = sentences
        values, indices, dense_shape = self.process_to_IDs_in_sparse_format(self.sp, messages)

        # Reduce logging output.
        tf.logging.set_verbosity(tf.logging.ERROR)
        
        with settings.MODEL_GRAPH.as_default():
            with settings.SESS.as_default() as session:
                # spm_path = sess.run(settings.MODULE(signature="spm_path"))
            # with tf.Session() as session:
                session.run([tf.global_variables_initializer(), tf.tables_initializer()])
                message_embeddings = session.run(
                settings.ENCODINGS,
                feed_dict={settings.INPUT_PLACEHOLDER.values: values,
                            settings.INPUT_PLACEHOLDER.indices: indices,
                            settings.INPUT_PLACEHOLDER.dense_shape: dense_shape})
        
        return message_embeddings


    def find_closest(self, sentence_rep,query_rep,K):
        top_K = np.argsort(np.sqrt((np.sum(np.square(sentence_rep - query_rep),axis=1))))[:K]
        return top_K

    def speak_like_me(self,query,K,your_embeddings,other_embeddings,your_sen):
        other_query = [query]
        query_embedding = self.embed_sentence_lite(other_query)
        closest_your = self.find_closest(your_embeddings,query_embedding,K)
        return [settings.YOUR_SENTENCES[c1].strip() for c1 in closest_your]
        # for cl in closest_your:
        #     print(settings.YOUR_SENTENCES[cl])


    def respond_like_me(self,query,K,key_embeddings,keys):
        other_query = [query]
        query_embedding = self.embed_sentence_lite(other_query)
        closest_other = self.find_closest(key_embeddings,query_embedding,K+2)
        return [settings.PR_TO_SP[keys[c1]].strip() for c1 in closest_other]
        # for k in closest_other[3:]:
        #     print(settings.PR_TO_SP[keys[k]])

    def post(self, request):
        serialzer = BotSerializer(data = request.data)
        if serialzer.is_valid():
            text = serialzer.validated_data.get('text')
            typee = serialzer.validated_data.get('type')
            size = serialzer.validated_data.get('size')
            response = {"error":"Not a valid type. Options: speak or reply"}
            if typee.lower() == 'speak':
                response = self.speak_like_me(text,size,settings.YOUR_EMBEDDINGS,settings.OTHER_EMBEDDINGS,settings.YOUR_SENTENCES)
            if typee.lower() == 'reply':
                response = self.respond_like_me(text,size,settings.KEY_EMBEDDINGS,settings.KEYS)
            return Response(response, status=200)
        else:
            error = {"error":', '.join(['{0}:{1}'.format(k, str(v[0])) for k, v in serialzer.errors.items()])}
            return Response(error, status=200)
