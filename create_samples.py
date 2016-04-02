from sampler import *
import os
from glob import glob
import tensorflow_wav

sampler = Sampler()

data = glob(os.path.join("./training", "*.wav"))
sample_file = data[0]
sample =tensorflow_wav.get_wav(sample_file)

z = sampler.generate_z()
for i in range(100):
    scale = i+1
    imagedata = sampler.generate(z, x_dim=64, y_dim=64, scale=scale)
    samplewav = sample.copy()
    samplewav['data']=np.dot(imagedata, 65535)

    filename = "samples/song-"+str(i)+".wav"
    tensorflow_wav.save_wav(samplewav, filename )
