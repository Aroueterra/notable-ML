#import tensorflow.compat.v1 as tf_v1
import tensorflow.compat.v1 as tf_v1
#To make tf 2.0 compatible with tf1.0 code, we disable the tf2.0 functionalities
#tf.disable_eager_execution()
tf_v1.compat.v1.disable_eager_execution()
import simpleaudio as sa
import numpy as np
from midi.player import *

import argparse
import tensorflow as tf
import ctc_utils
import cv2
import numpy as np


def load_pb(path_to_pb):
    print("load graph")
    with tf_v1.gfile.GFile(path_to_pb, "rb") as f:
        graph_def = tf_v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf_v1.Graph().as_default() as graph:
        tf_v1.import_graph_def(graph_def, name='')
        return graph

parser = argparse.ArgumentParser(description='Decode a music score image with a trained model (CTC).')
parser.add_argument('-image',  dest='image', type=str, required=True, help='Path to the input image.')
parser.add_argument('-model', dest='model', type=str, required=True, help='Path to the trained model.')
parser.add_argument('-vocabulary', dest='voc_file', type=str, required=True, help='Path to the vocabulary file.')
args = parser.parse_args()

tf_v1.reset_default_graph()
sess = tf_v1.InteractiveSession()
eta_path = r'C:\Users\aroue\Downloads\Documents\@ML\notable-ML\ML\models\frozen_model.pb'
# Read the dictionary
dict_file = open("Data/vocabulary_semantic.txt",'r')
dict_list = dict_file.read().splitlines()
int2word = dict()
for word in dict_list:
    word_idx = len(int2word)
    int2word[word_idx] = word
dict_file.close()

# Restore weights
#with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
#    graph_def = tf.GraphDef()
#    graph_def.ParseFromString(f.read())
#
#graph = tf.get_default_graph()
#tf.import_graph_def(graph_def, name="prefix")

#saver = tf_v1.train.import_meta_graph(args.model)
#saver.restore(sess,args.model[:-5])

#sess.run(tf.initialize_all_variables())

#graph = tf_v1.get_default_graph()

#with tf_v1.gfile.GFile(eta_path, "rb") as f:
#    graph_def = tf_v1.GraphDef()
#    graph_def.ParseFromString(f.read())
#    # import graph_def
#with tf_v1.Graph().as_default() as graph:
#    
#    tf_v1.import_graph_def(graph_def, name='') 
#graph = tf_v1.get_default_graph()    


print("load graph")

graph = load_pb(eta_path)

#with tf_v1.gfile.FastGFile(eta_path,'rb') as f:
#    graph_def = tf_v1.GraphDef()
#    graph_def.ParseFromString(f.read())
#    sess.graph.as_default()
#    tf_v1.import_graph_def(graph_def, name='')
#persisted_result = sess.graph.get_tensor_by_name("saved_result:0")
#tf.add_to_collection(tf.GraphKeys.VARIABLES,persisted_result)
#try:
    #saver = tf.train.Saver(tf.all_variables())
#except:pass
    #print("load data")
#saver.restore(sess, "C:/Users/aroue/Downloads/Documents/@ML/MODELS/tf-end-to-end/Models/semantic_model")  # now OK
    

input = graph.get_tensor_by_name("model_input:0")
seq_len = graph.get_tensor_by_name("seq_lengths:0")
rnn_keep_prob = graph.get_tensor_by_name("keep_prob:0")
height_tensor = graph.get_tensor_by_name("input_height:0")
width_reduction_tensor = graph.get_tensor_by_name("width_reduction:0")
printer = tf_v1.get_default_graph().get_all_collection_keys()
print(printer)
#logits = tf_v1.get_collection("logits")[0]

# Constants that are saved inside the model itself
WIDTH_REDUCTION, HEIGHT = sess.run([width_reduction_tensor, height_tensor])

decoded, _ = tf_v1.nn.ctc_greedy_decoder(logits, seq_len)

image = cv2.imread("Data/Example/000051652-1_2_1.png",0)
image = ctc_utils.resize(image, HEIGHT)
image = ctc_utils.normalize(image)
image = np.asarray(image).reshape(1,image.shape[0],-1,1)

seq_lengths = [ image.shape[2] / WIDTH_REDUCTION ]

prediction = sess.run(decoded,
                      feed_dict={
                          input: image,
                          seq_len: seq_lengths,
                          rnn_keep_prob: 1.0,
                      })

str_predictions = ctc_utils.sparse_tensor_to_strs(prediction)
for w in str_predictions[0]:
    print (int2word[w]),
    print ('\t'),

# form string of detected musical notes
SEMANTIC = ''
for w in str_predictions[0]:
    SEMANTIC += int2word[w] + '\n'

#with tf.Session(graph) as sess:
#    file_writer = tf.summary.FileWriter(logdir='C:/Users/aroue/Downloads/Documents/@ML/MODELS/tf-end-to-end/export_dir', graph=g)


    
if __name__ == '__main__':
    # gets the audio file
    audio = get_sinewave_audio(SEMANTIC)
    # horizontally stacks the freqs    
    audio =  np.hstack(audio)
    # normalizes the freqs
    audio *= 32767 / np.max(np.abs(audio))
    #converts it to 16 bits
    audio = audio.astype(np.int16)
    #plays midi 
    play_obj = sa.play_buffer(audio, 1, 2, 44100)
    #outputs to the console
    if play_obj.is_playing():
        print("\nplaying...")
        print(f'\n{SEMANTIC}')  
    #stop playback when done
    play_obj.wait_done()    