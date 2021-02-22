import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from tqdm import tqdm, tqdm_notebook
import random
import PIL
import time
from tqdm import *
import pdb
import requests
from PIL import Image
import numpy, numpy.fft
import shutil
import scipy, sklearn, librosa, urllib, IPython.display
import librosa.display

#import essentia, essentia.standard as ess
#plt.rcParams['figure.figsize'] = (14,4)

import torch
import torchvision

import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.utils import save_image

os.chdir('/content/drive/MyDrive/Adv-ML dataset')



os.chdir('/content/drive/MyDrive/Adv-ML dataset')

#!ls
face_data = 'vgg_face'
voice_data = 'vox_celeb'
face_names = os.listdir(face_data)
voice_names = os.listdir(voice_data)
names_file = pd.read_csv('vox_celeb_meta - Sheet1.csv')
Celeb_id = names_file['VoxCeleb1 ID']
VGG_name = names_file['VGGFace1 ID']





import string
def parse_metafile(meta_file):
    with open(meta_file, 'r') as f:
        lines = f.readlines()[1:]
    celeb_ids = {}
    for line in lines:
        ID, name, _, _, _ = line.rstrip().split(',')
        celeb_ids[ID] = name
    return celeb_ids
#parse_metafile('vox_celeb_meta - Sheet1.csv')


def get_labels(voice_list, face_list):
    voice_names = {item['name'] for item in voice_list}
    face_names = {item['name'] for item in face_list}
    names = voice_names & face_names

    voice_list = [item for item in voice_list if item['name'] in names]
    face_list = [item for item in face_list if item['name'] in names]

    names = sorted(list(names))
    label_dict = dict(zip(names, range(len(names))))
    for item in voice_list+face_list:
        item['label_id'] = label_dict[item['name']]
    return voice_list, face_list, len(names)
#get_labels

#celeb_ids = parse_metafile('vox_celeb_meta - Sheet1.csv')
#/content/drive/MyDrive/Adv-ML dataset/vgg_face
def get_dataset_files(data_dir, data_ext, celeb_ids, split):
    data_list = []
    # read data directory
    for root, dirs, filenames in os.walk(data_dir):
        for filename in filenames:
            if filename.endswith(data_ext):
                filepath = os.path.join(root, filename)
                # so hacky, be careful! 
                folder = filepath[len(data_dir):].split('/')[1]
                celeb_name = celeb_ids.get(folder, folder)
                if celeb_name.startswith(tuple(split)):
                    data_list.append({'filepath': filepath, 'name': celeb_name})
    return data_list 
#get_dataset_files('vox_celeb','.npy',celeb_ids,string.ascii_uppercase)

def get_dataset():
    celeb_ids = parse_metafile('vox_celeb_meta - Sheet1.csv')
    
    voice_list = get_dataset_files('vox_celeb',
                                   '.npy',
                                   celeb_ids,
                                   string.ascii_uppercase)
    face_list = get_dataset_files('vgg_face',
                                  '.jpg',
                                  celeb_ids,
                                  string.ascii_uppercase)
    return get_labels(voice_list, face_list)







import collections
import contextlib
import sys
import wave

def read_wave(path):
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def write_wave(path, audio, sample_rate):
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class Frame(object):
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    voiced_frames = []
    for frame in frames:
        #sys.stdout.write(
        #    '1' if vad.is_speech(frame.bytes, sample_rate) else '0')
        if not triggered:
            ring_buffer.append(frame)
            num_voiced = len([f for f in ring_buffer
                              if vad.is_speech(f.bytes, sample_rate)])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                # sys.stdout.write('+(%s)' % (ring_buffer[0].timestamp,))
                triggered = True
                voiced_frames.extend(ring_buffer)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append(frame)
            num_unvoiced = len([f for f in ring_buffer
                                if not vad.is_speech(f.bytes, sample_rate)])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                #sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])






import os
import torch
import shutil
import numpy as np
import torch.nn.functional as F

from PIL import Image
from scipy.io import wavfile
from torch.utils.data.dataloader import default_collate


class Meter(object):
    # Computes and stores the average and current value
    def __init__(self, name, display, fmt=':f'):
        self.name = name
        self.display = display
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name}:{' + self.display  + self.fmt + '},'
        return fmtstr.format(**self.__dict__)

def get_collate_fn(nframe_range):
    def collate_fn(batch):
        min_nframe, max_nframe = nframe_range
        assert min_nframe <= max_nframe
        num_frame = np.random.randint(min_nframe, max_nframe+1)
        pt = np.random.randint(0, max_nframe-num_frame+1)
        batch = [(item[0][..., pt:pt+num_frame], item[1])
                 for item in batch]
        return default_collate(batch)
    return collate_fn

def cycle(dataloader):
    while True:
        for data, label in dataloader:
            yield data, label

def save_model(net, model_path):
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
       os.makedirs(model_dir)
    torch.save(net.state_dict(), model_path)

def rm_sil(voice_file, vad_obj):
    """
       This code snippet is basically taken from the repository
           'https://github.com/wiseman/py-webrtcvad'

       It removes the silence clips in a speech recording
    """
    audio, sample_rate = read_wave(voice_file)
    frames = frame_generator(20, audio, sample_rate)
    frames = list(frames)
    segments = vad_collector(sample_rate, 20, 50, vad_obj, frames)

    if os.path.exists('tmp/'):
       shutil.rmtree('tmp/')
    os.makedirs('tmp/')

    wave_data = []
    for i, segment in enumerate(segments):
        segment_file = 'tmp/' + str(i) + '.wav'
        write_wave(segment_file, segment, sample_rate)
        wave_data.append(wavfile.read(segment_file)[1])
    shutil.rmtree('tmp/')

    if wave_data:
       vad_voice = np.concatenate(wave_data).astype('int16')
    return vad_voice

def get_fbank(voice, mfc_obj):
    # Extract log mel-spectrogra
    fbank = mfc_obj.sig2logspec(voice).astype('float32')

    # Mean and variance normalization of each mel-frequency 
    fbank = fbank - fbank.mean(axis=0)
    fbank = fbank / (fbank.std(axis=0)+np.finfo(np.float32).eps)

    # If the duration of a voice recording is less than 10 seconds (1000 frames),
    # repeat the recording until it is longer than 10 seconds and crop.
    full_frame_number = 1000
    init_frame_number = fbank.shape[0]
    while fbank.shape[0] < full_frame_number:
          fbank = np.append(fbank, fbank[0:init_frame_number], axis=0)
          fbank = fbank[0:full_frame_number,:]
    return fbank


def voice2face(e_net, g_net, voice_file, vad_obj, mfc_obj, GPU=True):
    vad_voice = rm_sil(voice_file, vad_obj)
    fbank = get_fbank(vad_voice, mfc_obj)
    fbank = fbank.T[np.newaxis, ...]
    fbank = torch.from_numpy(fbank.astype('float32'))
    
    if GPU:
        fbank = fbank.cuda()
    embedding = e_net(fbank)
    embedding = F.normalize(embedding)
    face = g_net(embedding)
    return face






!pip install webrtcvad
import pkg_resources

import _webrtcvad

__author__ = "John Wiseman jjwiseman@gmail.com"
__copyright__ = "Copyright (C) 2016 John Wiseman"
__license__ = "MIT"
__version__ = pkg_resources.get_distribution('webrtcvad').version


class Vad(object):
    def __init__(self, mode=None):
        self._vad = _webrtcvad.create()
        _webrtcvad.init(self._vad)
        if mode is not None:
            self.set_mode(mode)

    def set_mode(self, mode):
        _webrtcvad.set_mode(self._vad, mode)

    def is_speech(self, buf, sample_rate, length=None):
        length = length or int(len(buf) / 2)
        if length * 2 > len(buf):
            raise IndexError(
                'buffer has %s frames, but length argument was %s' % (
                    int(len(buf) / 2.0), length))
        return _webrtcvad.process(self._vad, sample_rate, buf, length)


def valid_rate_and_frame_length(rate, frame_length):
    return _webrtcvad.valid_rate_and_frame_length(rate, frame_length)







def load_voice(voice_item):
    voice_data = np.load(voice_item['filepath'])
    voice_data = voice_data.T.astype('float32')
    voice_label = voice_item['label_id']
    return voice_data, voice_label

def load_face(face_item):
    face_data = Image.open(face_item['filepath']).convert('RGB').resize([64, 64])
    face_data = np.transpose(np.array(face_data), (2, 0, 1))
    face_data = ((face_data - 127.5) / 127.5).astype('float32')
    face_label = face_item['label_id']
    return face_data, face_label

class VoiceDataset(Dataset):
    def __init__(self, voice_list, nframe_range):
        self.voice_list = voice_list
        self.crop_nframe = nframe_range[1]

    def __getitem__(self, index):
        voice_data, voice_label = load_voice(self.voice_list[index])
        assert self.crop_nframe <= voice_data.shape[1]
        pt = np.random.randint(voice_data.shape[1] - self.crop_nframe + 1)
        voice_data = voice_data[:, pt:pt+self.crop_nframe]
        return voice_data, voice_label

    def __len__(self):
        return len(self.voice_list)

class FaceDataset(Dataset):
    def __init__(self, face_list):
        self.face_list = face_list

    def __getitem__(self, index):
        face_data, face_label = load_face(self.face_list[index])
        if np.random.random() > 0.5:
           face_data = np.flip(face_data, axis=2).copy()
        return face_data, face_label

    def __len__(self):
        return len(self.face_list)






def count_parameters( model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)






class Voice_embedding(nn.Module):
    def __init__(self):
        super(Voice_embedding, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(64, 256, 3, 2, 1, bias=False),
            nn.BatchNorm1d(256, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 384, 3, 2, 1, bias=False),
            nn.BatchNorm1d(384, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(384, 576, 3, 2, 1, bias=False),
            nn.BatchNorm1d(576, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(576, 864, 3, 2, 1, bias=False),
            nn.BatchNorm1d(864, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(864, 64, 3, 2, 1, bias=True),
        )

    def forward(self, x):
        x = self.model(x)
        x = F.avg_pool1d(x, x.size()[2], stride=1)
        x = x.view(x.size()[0], -1, 1, 1)
        return x



#print(Voice_embedding())
param_voice = count_parameters(Voice_embedding())
print("Total number of parameters for Voice Embedding are: ", param_voice)


def get_network_voice(train=True):
    net = Voice_embedding()

    if True:
        net.cuda()

    if train:
        net.train()
        optimizer = Adam(net.parameters(),
                               lr=0.0002,
                               betas=(0.5, 0.999))
    else:
        """net.eval()
        net.load_state_dict(torch.load('./voice_embedding.pth'))"""
        optimizer = None
    return net, optimizer


net, opt = get_network_voice()




class Face_Embedding(nn.Module):
    def __init__(self):
        super(Face_Embedding, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 1, 1, 0, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 64, 4, 1, 0, bias=True),
        )
 
    def forward(self, x):
        x = self.model(x)
        return x



param_face = count_parameters(Face_Embedding())
print("Total number of parameters for Face Embedding are: ", param_face)



def get_network_face(train=True):
    net = Face_Embedding()

    if True:
        net.cuda()

    if train:
        net.train()
        optimizer = Adam(net.parameters(),
                               lr=0.0002,
                               betas=(0.5, 0.999))
    return net, optimizer






class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(64, 1024, 4, 1, 0, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, 1, 1, 0, bias=True),
        )
    def forward(self, x):
        x = self.model(x)
        return x


#print(Generator())
param_gen = count_parameters( Generator())
print("Total number of parameters for Generator are: ", param_gen)



def get_network_gen(train=True):
    net = Generator()

    if True:
        net.cuda()

    if train:
        net.train()
        optimizer = Adam(net.parameters(),
                               lr=0.0002,
                               betas=(0.5, 0.999))
    return net, optimizer




class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Linear(64, 1, bias=False)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.model(x)
        return x



def get_network_disc(train=True):
    net = Discriminator()

    if True:
        net.cuda()

    if train:
        net.train()
        optimizer = Adam(net.parameters(),
                               lr=0.0002,
                               betas=(0.5, 0.999))
    return net, optimizer




class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.model = nn.Linear(64, 59, bias=False)

    def forward(self, x):
        x = x.view(x.size()[0], -1)  ### Need to check once again of the class number
        x = self.model(x)
        return x



def get_network_cls(train=True):
    net = Classifier()

    if True:
        net.cuda()

    if train:
        net.train()
        optimizer = Adam(net.parameters(),
                               lr=0.0002,
                               betas=(0.5, 0.999))
    return net, optimizer






# dataset and dataloader
print('Parsing your dataset...')
voice_list, face_list, id_class_num = get_dataset()
#####################################################
#NETWORKS_PARAMETERS['c']['output_channel'] = 60 ### Need to check once

print('Preparing the datasets...')
voice_dataset = VoiceDataset(voice_list,[300, 800])
face_dataset = FaceDataset(face_list)

print('Dataset preparation Complete. Preparing the dataloaders...')
collate_fn = get_collate_fn([300, 800])
voice_loader = DataLoader(voice_dataset, shuffle=True, drop_last=True,
                          batch_size= 128,
                          num_workers= 1,
                          collate_fn=collate_fn)
face_loader = DataLoader(face_dataset, shuffle=True, drop_last=True,
                         batch_size= 128,
                         num_workers= 1)

voice_iterator = iter(cycle(voice_loader))
face_iterator = iter(cycle(face_loader))

# networks, Fe, Fg, Fd (f+d), Fc (f+c)
print('Dataloader prepared. Now, initializing networks...')
e_net, e_optimizer = get_network_voice(train = False)

g_net, g_optimizer = get_network_gen(train = True)
f_net, f_optimizer = get_network_face(train = True)
d_net, d_optimizer = get_network_disc(train = True)
c_net, c_optimizer = get_network_cls(train = True)

# label for real/fake faces
real_label = torch.full((128, 1), 1)   ### 128 is the batch size
fake_label = torch.full((128, 1), 0)

# Meters for recording the training status
iteration = Meter('Iter', 'sum', ':5d')
data_time = Meter('Data', 'sum', ':4.2f')
batch_time = Meter('Time', 'sum', ':4.2f')
D_real = Meter('D_real', 'avg', ':3.2f')
D_fake = Meter('D_fake', 'avg', ':3.2f')
C_real = Meter('C_real', 'avg', ':3.2f')
GD_fake = Meter('G_D_fake', 'avg', ':3.2f')
GC_fake = Meter('G_C_fake', 'avg', ':3.2f')
print('All done!!!')








disc_loss = []
gen_loss = []
for it in range(25000):
    start_time = time.time()
    
    voice, voice_label = next(voice_iterator)
    face, face_label = next(face_iterator)
    noise = 0.05*torch.randn(128, 64, 1, 1)

    if True:  # If I need GPU. Else, have to remove .cuda() parts.
        voice, voice_label = voice.cuda(), voice_label.cuda()
        face, face_label = face.cuda(), face_label.cuda()
        real_label, fake_label = real_label.cuda(), fake_label.cuda()
        noise = noise.cuda()
    data_time.update(time.time() - start_time)
    # Embeddings and generated faces
    embeddings = e_net(voice)
    embeddings = F.normalize(embeddings)
    # Here, Gaussian noise is added to the voice embeddings for generator part.
    embeddings = embeddings + noise
    embeddings = F.normalize(embeddings)
    fake = g_net(embeddings)

    # Discriminator updation
    f_optimizer.zero_grad()
    d_optimizer.zero_grad()
    c_optimizer.zero_grad()
    real_score_out = d_net(f_net(face))
    fake_score_out = d_net(f_net(fake.detach()))
    real_label_out = c_net(f_net(face))
    D_real_loss = F.binary_cross_entropy(torch.sigmoid(real_score_out), real_label.float())
    D_fake_loss = F.binary_cross_entropy(torch.sigmoid(fake_score_out), fake_label.float())
    C_real_loss = F.nll_loss(F.log_softmax(real_label_out, 1), face_label)
    D_real.update(D_real_loss.item())
    D_fake.update(D_fake_loss.item())
    C_real.update(C_real_loss.item())
    (D_real_loss + D_fake_loss + C_real_loss).backward()
    f_optimizer.step()
    d_optimizer.step()
    c_optimizer.step()
    disc_loss.append((D_real_loss + D_fake_loss + C_real_loss).item())

    # Generator
    g_optimizer.zero_grad()
    fake_score_out = d_net(f_net(fake))
    fake_label_out = c_net(f_net(fake))
    GD_fake_loss = F.binary_cross_entropy(torch.sigmoid(fake_score_out), real_label.float())
    GC_fake_loss = F.nll_loss(F.log_softmax(fake_label_out, 1), voice_label)
    (GD_fake_loss + GC_fake_loss).backward()
    GD_fake.update(GD_fake_loss.item())
    GC_fake.update(GC_fake_loss.item())
    g_optimizer.step()
    gen_loss.append((GD_fake_loss + GC_fake_loss).item())

    batch_time.update(time.time() - start_time)

    if it % 200 == 0:
        print(iteration, data_time, batch_time, 
              D_real, D_fake, C_real, GD_fake, GC_fake)
        data_time.reset()
        batch_time.reset()
        D_real.reset()
        D_fake.reset()
        C_real.reset()
        GD_fake.reset()
        GC_fake.reset()
        # snapshot
        save_model(g_net, '/content/drive/MyDrive/Adv-ML dataset/my_model/classifier.pth')   ### Need to update location!!!
    iteration.update(1)
