import os
import json
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm
import argparse
import re
import shutil
from pathlib import Path

# Import our English frontend
from modules.english_frontend import EnglishFrontend

class EnglishDataProcessor:
    def __init__(self, config):
        self.config = config
        self.raw_data_dir = config.raw_data_dir
        self.processed_data_dir = config.processed_data_dir
        self.sample_rate = config.audio_sample_rate
        
        # Initialize the English frontend
        self.text_frontend = EnglishFrontend(config)
        
        # Create necessary directories
        os.makedirs(self.processed_data_dir, exist_ok=True)
        os.makedirs(f"{self.processed_data_dir}/wavs", exist_ok=True)
        
    def _normalize_text(self, text):
        """Clean and normalize the text"""
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def process_transcript(self, transcript_file):
        """Process a transcript file containing text-audio pairs"""
        all_items = []
        
        print(f"Processing transcript: {transcript_file}")
        with open(transcript_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                parts = line.strip().split('|')
                if len(parts) < 2:
                    print(f"Skipping invalid line: {line}")
                    continue
                
                wav_path = parts[0].strip()
                text = parts[1].strip()
                
                # Skip if text is empty
                if not text:
                    print(f"Skipping empty text for {wav_path}")
                    continue
                
                # Get full path to audio file
                if not os.path.isabs(wav_path):
                    wav_path = os.path.join(os.path.dirname(transcript_file), wav_path)
                
                # Skip if audio file doesn't exist
                if not os.path.exists(wav_path):
                    print(f"Audio file not found: {wav_path}")
                    continue
                
                # Load and resample audio
                try:
                    y, sr = librosa.load(wav_path, sr=self.sample_rate)
                    
                    # Skip very short audios
                    if len(y) < self.sample_rate * 0.5:  # shorter than 0.5s
                        print(f"Audio too short: {wav_path}")
                        continue
                    
                    # Normalize audio
                    y = y / np.max(np.abs(y)) * 0.95
                    
                    # Generate unique ID for this item
                    item_id = f"item_{len(all_items):08d}"
                    out_wav_path = f"{self.processed_data_dir}/wavs/{item_id}.wav"
                    
                    # Save normalized audio
                    sf.write(out_wav_path, y, self.sample_rate)
                    
                    # Process text
                    text = self._normalize_text(text)
                    phonemes = self.text_frontend.get_phoneme_sequence(text)
                    
                    # Create item
                    item = {
                        'id': item_id,
                        'text': text,
                        'phonemes': phonemes,
                        'wav_path': out_wav_path,
                        'wav_duration': len(y) / self.sample_rate
                    }
                    
                    all_items.append(item)
                except Exception as e:
                    print(f"Error processing {wav_path}: {e}")
        
        return all_items
    
    def process_dataset(self):
        """Process the entire dataset"""
        transcript_files = []
        
        # Find all transcript files
        for root, _, files in os.walk(self.raw_data_dir):
            for file in files:
                if file.endswith('.txt') or file.endswith('.trans'):
                    transcript_files.append(os.path.join(root, file))
        
        if not transcript_files:
            raise ValueError(f"No transcript files found in {self.raw_data_dir}")
        
        # Process all transcripts
        all_items = []
        for transcript_file in transcript_files:
            items = self.process_transcript(transcript_file)
            all_items.extend(items)
        
        # Create train/val/test splits
        np.random.shuffle(all_items)
        val_size = min(int(len(all_items) * 0.1), 100)
        test_size = min(int(len(all_items) * 0.05), 50)
        
        train_items = all_items[:(len(all_items) - val_size - test_size)]
        val_items = all_items[(len(all_items) - val_size - test_size):(len(all_items) - test_size)]
        test_items = all_items[(len(all_items) - test_size):]
        
        # Save splits
        with open(f"{self.processed_data_dir}/train.json", 'w') as f:
            json.dump(train_items, f, indent=2)
        
        with open(f"{self.processed_data_dir}/val.json", 'w') as f:
            json.dump(val_items, f, indent=2)
        
        with open(f"{self.processed_data_dir}/test.json", 'w') as f:
            json.dump(test_items, f, indent=2)
        
        print(f"Dataset processed successfully. Total items: {len(all_items)}")
        print(f"Train: {len(train_items)}, Val: {len(val_items)}, Test: {len(test_items)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='base', help='config name')
    args = parser.parse_args()
    
    from english_dataset_config import get_config
    config = get_config(args.config)
    
    processor = EnglishDataProcessor(config)
    processor.process_dataset()

if __name__ == '__main__':
    main()
