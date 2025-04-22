import os
import json
import random
import numpy as np
import torch
import librosa
from torch.utils.data import Dataset

from utils.commons.mel_processing import mel_spectrogram_torch
from modules.english_frontend import EnglishFrontend
from utils.commons.indexed_datasets import IndexedDataset

class EnglishDiffSingerDataset(Dataset):
    def __init__(self, config, data_dir, split='train'):
        super().__init__()
        self.config = config
        self.data_dir = data_dir
        self.split = split
        
        # Load data
        with open(f"{data_dir}/{split}.json", 'r') as f:
            self.data = json.load(f)
            
        # Initialize text frontend
        self.text_frontend = EnglishFrontend(config)
        
        # Audio parameters
        self.sample_rate = config.audio_sample_rate
        self.hop_size = config.hop_size
        self.fft_size = config.fft_size
        self.win_size = config.win_size
        self.num_mels = config.audio_num_mel_bins
        
        # Load phoneme dictionary
        phoneme_dict_path = os.path.join(os.path.dirname(__file__), config.phoneme_dict_file)
        self.phoneme_dict = {}
        with open(phoneme_dict_path, 'r') as f:
            exec(f.read(), self.phoneme_dict)
        
        # For training
        self.max_frames = config.max_frames
        self.max_input_tokens = config.max_input_tokens
        
        print(f"Loaded {len(self.data)} items for {split}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        wav_path = item['wav_path']
        text = item['text']
        phonemes = item['phonemes'].split()
        
        # Load audio
        audio, _ = librosa.load(wav_path, sr=self.sample_rate)
        
        # Compute mel spectrogram
        mel = self._get_mel(audio)
        
        # Convert phonemes to IDs
        phoneme_ids = []
        for p in phonemes:
            if p in self.phoneme_dict['phoneme_to_id']:
                phoneme_ids.append(self.phoneme_dict['phoneme_to_id'][p])
            else:
                # Skip unknown phonemes
                continue
        
        # Add EOS token
        phoneme_ids.append(self.phoneme_dict['phoneme_to_id'][self.phoneme_dict['eos']])
        
        # Create input tensor
        phoneme_ids = torch.LongTensor(phoneme_ids)
        
        # For training, we may need to limit the length
        if self.split == 'train':
            if len(phoneme_ids) > self.max_input_tokens:
                # Truncate input
                phoneme_ids = phoneme_ids[:self.max_input_tokens]
            
            if mel.shape[0] > self.max_frames:
                # Random crop
                start = random.randint(0, mel.shape[0] - self.max_frames)
                mel = mel[start:start + self.max_frames]
        
        return {
            'id': item['id'],
            'text': text,
            'phonemes': item['phonemes'],
            'phoneme_ids': phoneme_ids,
            'mel': mel,
            'mel_lengths': mel.shape[0],
            'phoneme_lengths': len(phoneme_ids),
        }
    
    def _get_mel(self, audio):
        """Convert audio to mel spectrogram"""
        audio = torch.FloatTensor(audio).unsqueeze(0)
        mel = mel_spectrogram_torch(
            audio, 
            self.fft_size,
            self.num_mels, 
            self.sample_rate,
            self.hop_size,
            self.win_size,
            self.config.fmin,
            self.config.fmax,
            center=True
        ).squeeze(0).numpy()
        return mel
    
    def collate_fn(self, batch):
        """Collate function for DataLoader"""
        # Sort by input length for packing
        batch.sort(key=lambda x: x['phoneme_lengths'], reverse=True)
        
        # Get max lengths
        max_phoneme_len = max([x['phoneme_lengths'] for x in batch])
        max_mel_len = max([x['mel_lengths'] for x in batch])
        
        # Initialize tensors
        phoneme_ids = torch.zeros(len(batch), max_phoneme_len).long()
        phoneme_lengths = torch.zeros(len(batch)).long()
        mels = torch.zeros(len(batch), max_mel_len, self.num_mels)
        mel_lengths = torch.zeros(len(batch)).long()
        
        # Fill tensors
        for i, item in enumerate(batch):
            phoneme_ids[i, :item['phoneme_lengths']] = item['phoneme_ids']
            phoneme_lengths[i] = item['phoneme_lengths']
            mels[i, :item['mel_lengths']] = torch.FloatTensor(item['mel'])
            mel_lengths[i] = item['mel_lengths']
        
        return {
            'ids': [x['id'] for x in batch],
            'texts': [x['text'] for x in batch],
            'phonemes': [x['phonemes'] for x in batch],
            'phoneme_ids': phoneme_ids,
            'phoneme_lengths': phoneme_lengths,
            'mels': mels,
            'mel_lengths': mel_lengths,
        }
