import os

from utils.hparams import hparams, set_hparams

# Base config overrides
base_config = {
    "task_cls": "tasks.tts.diffsinger.DiffSinger",
    "vocoder_ckpt": "./checkpoints/hifigan/g_00000000",
    "work_dir": "./checkpoints/english_diffsinger",
    
    # Data configurations
    "raw_data_dir": "./data/raw/english_corpus",  # Path to your English corpus
    "processed_data_dir": "./data/processed/english_diffsinger",
    
    # Audio configurations
    "audio_sample_rate": 22050,
    "hop_size": 256,
    "fft_size": 1024,
    "win_size": 1024,
    "audio_num_mel_bins": 80,
    "fmin": 0,
    "fmax": 8000,
    
    # Phoneme configurations
    "use_english_frontend": True,
    "phoneme_dict_file": "english_phoneme_dict.py",
    
    # Model configurations
    "hidden_size": 256,
    "enc_layers": 4,
    "dec_layers": 4,
    "enc_ffn_kernel_size": 9,
    "dec_ffn_kernel_size": 9,
    
    # Diffusion configurations
    "use_shallow_diffusion": True,
    "timesteps": 1000,
    "pe_scale": 1000,
    
    # Training configurations
    "max_frames": 1000,
    "max_input_tokens": 200,
    "batch_size": 16,
    "valid_batch_size": 4,
    "val_check_interval": 2000,
    "max_updates": 160000,
    "lr": 2e-4,
    "lr_decay": 0.999,
}

class EnglishDiffSingerConfig:
    def __init__(self):
        for k, v in base_config.items():
            setattr(self, k, v)
        
    def update(self, config_dict):
        for k, v in config_dict.items():
            setattr(self, k, v)

# Create configs for different scenarios
def get_config(config_name='base'):
    config = EnglishDiffSingerConfig()
    
    if config_name == 'base':
        return config
    
    if config_name == 'small':
        config.update({
            "hidden_size": 192,
            "enc_layers": 3,
            "dec_layers": 3,
            "batch_size": 8,
        })
        return config
    
    if config_name == 'large':
        config.update({
            "hidden_size": 384,
            "enc_layers": 6,
            "dec_layers": 6,
            "batch_size": 8,
        })
        return config
    
    return config
