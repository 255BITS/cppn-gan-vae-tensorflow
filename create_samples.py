import os
from glob import glob
import tensorflow as tf
import numpy as np
import tensorflow_wav
from model import CPPNVAE

model = CPPNVAE(batch_size=1)
model.load_model('save')
print("Loaded")

data = glob(os.path.join("./training", "*.wav"))
sample_file = data[0]
sample =tensorflow_wav.get_wav(sample_file)

def new_z():
  return np.random.normal(size=(1, 32)).astype(np.float32)
print("Generating")

bitrate = 4096
t_dims = [bitrate, bitrate*2]
for t_dim in t_dims:
    wavdata = [model.generate(new_z(), t_dim=t_dim) for i in range(10)]
    samplewav = sample.copy()
    wavdata = np.reshape(wavdata, [-1])
    print(np.array(wavdata), "/before")

    #biggest = np.max(np.abs(np.max(wavdata)), np.abs(np.min(wavdata)))
    print("Biggest", np.max(wavdata))
    samplewav['data']=wavdata

    filename = "samples/song_t"+str(t_dim)+".wav"
    tensorflow_wav.save_wav(samplewav, filename )
