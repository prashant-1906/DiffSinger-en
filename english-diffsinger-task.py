import torch
import numpy as np
from tasks.tts.diffsinger import DiffSinger as BaseDiffSinger
from modules.english_frontend import EnglishFrontend
from utils.commons.hparams import hparams

class EnglishDiffSinger(BaseDiffSinger):
    def __init__(self, config=None):
        # Initialize with the base DiffSinger implementation
        super().__init__(config)
        
        # Override the text frontend with our English implementation
        self.text_frontend = EnglishFrontend(config)
        
        # Load English phoneme dictionary
        self._load_phoneme_dict()
        
    def _load_phoneme_dict(self):
        """Load phoneme dictionary from the config"""
        import importlib.util
        import sys
        import os
        
        # Load phoneme dict
        phoneme_dict_path = os.path.join(os.path.dirname(__file__), self.config.phoneme_dict_file)
        spec = importlib.util.spec_from_file_location("phoneme_dict", phoneme_dict_path)
        phoneme_dict_module = importlib.util.module_from_spec(spec)
        sys.modules["phoneme_dict"] = phoneme_dict_module
        spec.loader.exec_module(phoneme_dict_module)
        
        # Set phoneme dictionary
        self.phone_dict = phoneme_dict_module.phoneme_dict
        
        # Update model config
        self.model.n_vocab = len(self.phone_dict['phoneme_set'])
    
    def preprocess_input(self, text):
        """Preprocess text input for inference"""
        # Get phoneme sequence
        phoneme_sequence = self.text_frontend.get_phoneme_sequence(text)
        phoneme_ids = []
        
        # Convert to IDs
        for phoneme in phoneme_sequence.split():
            if phoneme in self.phone_dict['phoneme_to_id']:
                phoneme_ids.append(self.phone_dict['phoneme_to_id'][phoneme])
        
        # Add EOS token
        phoneme_ids.append(self.phone_dict['phoneme_to_id'][self.phone_dict['eos']])
        
        # Create tensor
        phoneme_ids = torch.LongTensor(phoneme_ids).unsqueeze(0)
        phoneme_lengths = torch.LongTensor([len(phoneme_ids[0])])
        
        return {
            'text': text,
            'phonemes': phoneme_sequence,
            'phoneme_ids': phoneme_ids,
            'phoneme_lengths': phoneme_lengths
        }
    
    def inference(self, text, **kwargs):
        """Run inference with text input"""
        self.model.eval()
        
        with torch.no_grad():
            # Process input
            processed_input = self.preprocess_input(text)
            
            # Move to device
            for k, v in processed_input.items():
                if isinstance(v, torch.Tensor):
                    processed_input[k] = v.to(self.device)
            
            # Run model inference
            output = self.model.inference(
                processed_input['phoneme_ids'],
                processed_input['phoneme_lengths'],
                **kwargs
            )
            
            # Get mel spectrogram
            mel_pred = output['mel_pred'][0].cpu().numpy()
            
            # Generate waveform using vocoder
            if hasattr(self, 'vocoder'):
                wav_pred = self.vocoder.spec2wav(mel_pred)
            else:
                wav_pred = None
            
            return {
                'mel_pred': mel_pred,
                'wav_pred': wav_pred
            }
    
    def test_step(self, sample, batch_idx):
        """Override test step for evaluation"""
        # Move to device
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                sample[k] = v.to(self.device)
        
        # Run model
        outputs = self.model(sample)
        
        # Compute metrics
        loss = outputs['loss']
        
        # Log metrics
        self.log('test_loss', loss.item(), prog_bar=True, sync_dist=True)
        
        return {'loss': loss.item()}
