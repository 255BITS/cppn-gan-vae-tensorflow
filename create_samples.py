import os
from glob import glob
import tensorflow as tf
import numpy as np
import tensorflow_wav
from model import CPPNVAE

model = CPPNVAE(batch_size=128)
model.load_model('save')

data = glob(os.path.join("./training", "*.wav"))
sample_file = data[0]
sample =tensorflow_wav.get_wav(sample_file)

z = np.random.normal(size=(128, 32)).astype(np.float32)
scale=1.0
print("Generating")
wavdata = model.generate(z, x_dim=64, y_dim=64, scale=scale)
print("Generated", np.shape(wavdata))
samplewav = sample.copy()
wavdata = np.reshape(wavdata, [-1])
wavdata = np.dot(wavdata, 32767)
samplewav['data']=wavdata
print(np.array(samplewav['data']))

filename = "samples/song2.wav"
tensorflow_wav.save_wav(samplewav, filename )
