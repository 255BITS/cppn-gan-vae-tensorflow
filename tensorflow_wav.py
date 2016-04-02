import wave
import numpy as np
import tensorflow as tf
from scipy.io.wavfile import read, write
import scipy
import ops
import math
import pickle

FRAME_SIZE=(64/2048)
HOP=(2048-64)/(2048*64)

def do_imdct(row):
    return mdct.imdct(row, len(row))


def convert_mlaudio_to_wav(mlaudio, dimensions, wav_x):
    audio = np.array(mlaudio['data'])
    audio = np.reshape(audio,[-1, dimensions])
    audio = audio[:,0].tolist()
    imdct_data = np.array(audio).reshape([-1, wav_x])
    imdct_data = [do_imdct(row) for row in imdct_data]

    print("NEW SHAPE", np.shape(imdct_data))
    mlaudio['data'] = np.array(imdct_data)
    return mlaudio


# Returns the file object in complex64
def get_wav(path):

    wav = wave.open(path, 'rb')
    rate, data = read(path)
    results={}
    results['rate']=rate
    results['channels']=wav.getnchannels()
    results['sampwidth']=wav.getsampwidth()
    results['framerate']=wav.getframerate()
    results['nframes']=wav.getnframes()
    results['compname']=wav.getcompname()
    processed = np.array(data).astype(np.int16, copy=False)
    results['data']=processed
    return results

def save_wav(in_wav, path):

    print("Saving to ", path)
    wav = wave.open(path, 'wb')
    wav.setnchannels(in_wav['channels'])
    wav.setsampwidth(in_wav['sampwidth'])

    wav.setframerate(in_wav['framerate'])

    wav.setnframes(in_wav['nframes'])

    wav.setcomptype('NONE', 'processed')

    processed = np.array(in_wav['data'], dtype=np.int16)
    wav.writeframes(processed)

def save_pre(in_wav, path):
    f = open(path, "wb")
    try:
        pickle.dump(in_wav, f, pickle.HIGHEST_PROTOCOL)
        print("DUMPED")
    finally:
        f.close()
def get_pre(filename):
    f = open(filename, "rb")
    data = pickle.load(f)
    f.close()
    return data


def compose(input, rank=3):
    return input

def encode(input,bitrate=4096):
    output = input

    output = tf.reshape(output, [-1, 64,64,1])
    output = compose(output)
    return output

def ff_nn(input, name):
    with tf.variable_scope("ff_nn"):
        input_shape = input.get_shape() 
        input_dim = int(input.get_shape()[1])
        W = tf.get_variable(name+'w',[input_dim, input_dim], initializer=tf.random_normal_initializer(0, 0.02))

        # Initialize b to zero
        b = tf.get_variable(name+'b', [input_dim], initializer=tf.constant_initializer(0))

        output = tf.nn.tanh(tf.matmul(tf.reshape(input, [-1,input_dim]),W) + b)
        output = tf.reshape(output, input_shape)
        return output



def scale_up(input):
    with tf.variable_scope("scale"):

        output = tf.nn.tanh(input)
        w = tf.get_variable('scale_w', [1], dtype=tf.float32, initializer=tf.constant_initializer(0.001))
        return output/w

        raw_output, fft_real_output= tf.split(3, 2, output)
        sign = tf.sign(raw_output)

        #raw = tf.exp(tf.abs(10.8*raw_output))*sign
        raw = raw_output * 32768

        #new_fft = ff_nn(fft_output, 'fft')
        #complex = tf.complex(fft_real_output, fft_imag_output)
        #fft = complex / w
        fft = fft_real_output / w

        return tf.concat(3, [raw, fft])
