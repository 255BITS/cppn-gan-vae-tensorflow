from sampler import *

sampler = Sampler()

z = sampler.generate_z()
for i in range(100):
    scale = i+1
    imagedata = sampler.generate(z, x_dim=64, y_dim=64, scale=scale)
    sampler.save_png(imagedata, "samples/sample-"+str(scale)+".wav")
