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

x_dims = [64, 128, 256, 512]
scales = [1.0, 2.0, 10.0]
y_dims = [64, 128]
for x_dim in x_dims:
    for y_dim in y_dims:
        for scale in scales:
            wavdata = [model.generate(new_z(), x_dim=x_dim, y_dim=y_dim, scale=scale) for i in range(10)]
            samplewav = sample.copy()
            wavdata = np.reshape(wavdata, [-1])
            print(np.array(wavdata), "/before")

            #biggest = np.max(np.abs(np.max(wavdata)), np.abs(np.min(wavdata)))
            wavdata = np.sin(wavdata * np.pi/2.0) * 32767.0
            print("Biggest", np.max(wavdata))
            samplewav['data']=wavdata

            filename = "samples/song_x"+str(x_dim)+"_y"+str(y_dim)+"_s"+str(scale)+".wav"
            tensorflow_wav.save_wav(samplewav, filename )
