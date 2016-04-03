import numpy as np
import tensorflow as tf

from glob import glob
import tensorflow_wav
import argparse
import time
import os

from model import CPPNVAE

'''
cppn vae:

compositional pattern-producing generative adversarial network

LOADS of help was taken from:

https://github.com/carpedm20/DCGAN-tensorflow
https://jmetzen.github.io/2015-11-27/vae.html

'''

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--training_epochs', type=int, default=100,
                     help='training epochs')
  parser.add_argument('--display_step', type=int, default=1,
                     help='display step')
  parser.add_argument('--checkpoint_step', type=int, default=1,
                     help='checkpoint step')
  parser.add_argument('--batch_size', type=int, default=128,
                     help='batch size')
  parser.add_argument('--learning_rate', type=float, default=0.002,
                     help='learning rate for G and VAE')
  parser.add_argument('--learning_rate_vae', type=float, default=0.001,
                     help='learning rate for VAE')
  parser.add_argument('--learning_rate_d', type=float, default=0.0005,
                     help='learning rate for D')
  parser.add_argument('--keep_prob', type=float, default=0.6,
                     help='dropout keep probability')
  parser.add_argument('--beta1', type=float, default=0.65,
                     help='adam momentum param for descriminator')
  args = parser.parse_args()
  return train(args)

def train(args):

  learning_rate = args.learning_rate
  learning_rate_d = args.learning_rate_d
  learning_rate_vae = args.learning_rate_vae
  batch_size = args.batch_size
  training_epochs = args.training_epochs
  display_step = args.display_step
  checkpoint_step = args.checkpoint_step # save training results every check point step
  beta1 = args.beta1
  keep_prob = args.keep_prob
  dirname = 'save'
  if not os.path.exists(dirname):
    os.makedirs(dirname)


  cppnvae = CPPNVAE(batch_size=batch_size, learning_rate = learning_rate, learning_rate_d = learning_rate_d, learning_rate_vae = learning_rate_vae, beta1 = beta1, keep_prob = keep_prob)

  # load previously trained model if appilcabl
  ckpt = tf.train.get_checkpoint_state(dirname)
  if ckpt:
    cppnvae.load_model(dirname)

  counter = 0
  loss_counter = 0

  # Training cycle
  for epoch in range(training_epochs):
    batch_files = glob(os.path.join("./training", "*.wav"))
    np.random.shuffle(batch_files)
    avg_d_loss = 0.
    avg_q_loss = 0.
    avg_vae_loss = 0.
    def get_wav_content(files):
        for filee in files:
            print("Yielding ", filee)
            try:
                yield tensorflow_wav.get_wav(filee)
            except Exception as e:
                print("Could not load ", filee, e)


    for filee in get_wav_content(batch_files):
      data = filee["data"]
      print("min", np.min(data), np.max(data))

      n_samples = len(data)
      samples_per_batch=batch_size * cppnvae.t_dim
      total_batch = int(n_samples / samples_per_batch)
        
      nonzero_data = []

      for i in range(total_batch):
          batch_audio =data[(i*samples_per_batch):((i+1)*samples_per_batch)]
          maxi = np.max(batch_audio)
          if(maxi > 1000):
            # toss batches with low volume
            nonzero_data = np.hstack((nonzero_data, batch_audio))
        
      print("Nonzero", len(nonzero_data), "All", len(data))
      n_samples = len(nonzero_data)
      total_batch = int(n_samples / samples_per_batch)



      try:

        # Loop over all batches
        for i in range(total_batch):
          batch_audio =data[(i*samples_per_batch):((i+1)*samples_per_batch)]
          batch_audio = np.reshape(batch_audio, (batch_size, cppnvae.t_dim, 1))
          batch_audio = np.dot(np.array(batch_audio, np.float32), 1.0/32767)
  
          d_loss, g_loss, vae_loss, n_operations = cppnvae.partial_train(batch_audio)
  
  
          # Display logs per epoch step
          if (counter+1) % display_step == 0:
            print("Sample:", '%d' % ((i+1)*batch_size), " Epoch:", '%d' % (epoch), \
                  "d_loss=", "{:.4f}".format(d_loss), \
                  "g_loss=", "{:.4f}".format(g_loss), \
                  "vae_loss=", "{:.4f}".format(vae_loss), \
                  "n_op=", '%d' % (n_operations))
  
          if(d_loss > 5 or g_loss > 5):
              loss_counter+=1
          else:
              loss_counter=0

          assert( vae_loss < 1000000 ) # make sure it is not NaN or Inf
          assert( d_loss < 1000000 ) # make sure it is not NaN or Inf
          assert( g_loss < 1000000 ) # make sure it is not NaN or Inf
          assert( loss_counter < 50 )

          counter += 1
          # Compute average loss
          avg_d_loss += d_loss / n_samples * batch_size
          avg_q_loss += g_loss / n_samples * batch_size
          avg_vae_loss += vae_loss / n_samples * batch_size
          
          if(counter % 60 == 0):
            if(g_loss > 5 or d_loss > 5):
                print("Refusing to save b/c loss is too high")
            else:
                # save model
                checkpoint_path = os.path.join('save', 'model.ckpt')
                cppnvae.save_model(checkpoint_path, epoch)
                print("model saved to {}".format(checkpoint_path))


      except:
          loss_counter = 0
          counter = 0
          print("Oh shit we diverged. Reloading and retrying different file")
          # load previously trained model if appilcabl
          ckpt = tf.train.get_checkpoint_state(dirname)
          if ckpt:
            cppnvae.load_model(dirname)

      finally:
          print("Next Song")
    # Display logs per epoch step
    if epoch >= 0:
      print("Epoch:", '%04d' % (epoch), \
            "avg_d_loss=", "{:.6f}".format(avg_d_loss), \
            "avg_q_loss=", "{:.6f}".format(avg_q_loss), \
            "avg_vae_loss=", "{:.6f}".format(avg_vae_loss))


  # save model one last time, under zero label to denote finish.
  cppnvae.save_model(checkpoint_path, 0)

if __name__ == '__main__':
  main()
