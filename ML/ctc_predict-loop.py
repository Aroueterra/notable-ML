#To make tf 2.0 compatible with tf1.0 code, we disable the tf2.0 functionalities
#tf.disable_eager_execution()
import os
import cv2
import io
import logging
import argparse
import ctc_utils
import numpy as np
import tensorflow as tf
import simpleaudio as sa
from pathlib import Path

from midi.player import *
import tensorflow.compat.v1 as tf_v1
from scipy.io.wavfile import write as WAV
import tensorflow.python.util.deprecation as deprecation
tf.get_logger().setLevel('FATAL')
tf.autograph.set_verbosity(1)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf_v1.compat.v1.disable_eager_execution()
deprecation._PRINT_DEPRECATION_WARNINGS = False
logging.getLogger('tensorflow').setLevel(logging.FATAL)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


parser = argparse.ArgumentParser(description='Decode a music score image with a trained model (CTC).')nargs='?',
parser.add_argument('-image',  dest='image', type=str, required=True, help='Path to the input image.')
parser.add_argument('-model', dest='model', type=str, required=True, help='Path to the trained model.')
parser.add_argument('-vocabulary', dest='voc_file', type=str, required=True, help='Path to the vocabulary file.')
parser.add_argument('-type',  dest='type', type=str, nargs='?', help='Path to the output type.')
args = parser.parse_args()

tf_v1.reset_default_graph()
sess = tf_v1.InteractiveSession()

# Read the dictionary
dict_file = open(args.voc_file,'r')
dict_list = dict_file.read().splitlines()
int2word = dict()
for word in dict_list:
    word_idx = len(int2word)
    int2word[word_idx] = word
dict_file.close()

# Restore weights
saver = tf_v1.train.import_meta_graph(args.model)
saver.restore(sess,args.model[:-5])

graph = tf_v1.get_default_graph()

input = graph.get_tensor_by_name("model_input:0")
seq_len = graph.get_tensor_by_name("seq_lengths:0")
rnn_keep_prob = graph.get_tensor_by_name("keep_prob:0")
height_tensor = graph.get_tensor_by_name("input_height:0")
width_reduction_tensor = graph.get_tensor_by_name("width_reduction:0")
logits = graph.get_tensor_by_name("fully_connected/BiasAdd:0")
#logits = tf_v1.get_collection("logits")[0]

# Constants that are saved inside the model itself
WIDTH_REDUCTION, HEIGHT = sess.run([width_reduction_tensor, height_tensor])

decoded, _ = tf_v1.nn.ctc_greedy_decoder(logits, seq_len)

path = Path(__file__).parent.absolute()

mypath = Path().absolute()
file_path = str(mypath) + '\\'
file_forward = Path(args.image)
absolute_path = Path(file_path + args.image)
absolute_str = str(absolute_path)
file_name = file_forward.name.split('.')[-2]
file_ext = str(absolute_path).split('.')[1]
#flen = len(file_parts)
counter = 1
all_predictions=[]
bassclef= ['clef-F3','clef-F4','clef-F5']
print("absolute initial " + str(mypath))
print("File name " + file_name)
print("File ext " + file_ext)
print(str(absolute_path))
print(absolute_path.is_file())
while absolute_path.exists():
    file_name = absolute_str.split('.')[-2]
    image = cv2.imread(str(absolute_path),0)
    image = ctc_utils.resize(image, 128)
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
    parsed_predictions = ''
    for w in str_predictions[0]:
        parsed_predictions += int2word[w] + '\n' 
    absolute_path = Path(file_name[:-1] + str(counter) + '.' + file_ext)
    counter+=1
    #check for bass clef matches and discard result
#     matching = [s for s in SEMANTIC if any(xs in s for xs in bassclef)]
#     print ("match? " + matching)
#     if matching:
#         continue
    all_predictions.append(parsed_predictions)
    
len(all_predictions)
    
if __name__ == '__main__':
    SEMANTIC = ''
    playlist = []
    track = 0
    export = 0
    directory=''
    if (args.type == "clean"):
        directory = 'Data\\clean\\'
    elif(args.type == "raw"):
        directory = 'Data\\raw\\'
    else:
        directory = 'Data\\perfect\\'
            
    for SEMANTIC in all_predictions:
        # gets the audio file
        audio = get_sinewave_audio(SEMANTIC)
        # horizontally stacks the freqs    
        audio =  np.hstack(audio)
        # normalizes the freqs
        audio *= 32767 / np.max(np.abs(audio))
        #converts it to 16 bits
        audio = audio.astype(np.int16)
        playlist.append(audio)
        
        print("added one song to playlist")
        
            
        
        with open(directory + 'predictions'+ str(export) +'.txt', 'w') as file:
            file.write(SEMANTIC)
        export+=1

    len(playlist)
    for song in playlist:
        output_file = directory + 'staff' + str(track) + '.wav'
        WAV(output_file, 44100, song)
        print("created wav file")
        track+=1
        #play_obj = sa.play_buffer(song, 1, 2, 44100)
        #outputs to the console
        #if play_obj.is_playing():
        #    print("\nplaying..." + str(track))
        #    print(f'\n{SEMANTIC}')  
        #stop playback when done
        #play_obj.wait_done()