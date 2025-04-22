import os
import argparse
import torch
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from english_diffsinger_task import EnglishDiffSinger
from english_dataset_config import get_config

def plot_mel(mel, path):
    """Plot mel spectrogram and save to path"""
    plt.figure(figsize=(10, 6))
    plt.imshow(mel.T, aspect='auto', origin='lower')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='base', help='config name')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to checkpoint')
    parser.add_argument('--input', type=str, default=None, help='input text')
    parser.add_argument('--input_file', type=str, default=None, help='input text file')
    parser.add_argument('--output_dir', type=str, default='./inference_output', help='output directory')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature for sampling')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load config
    config = get_config(args.config)
    
    # Load model
    model = EnglishDiffSinger(config)
    model.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    print(f"Loaded checkpoint from {args.checkpoint}")
    
    # Get input texts
    texts = []
    if args.input:
        texts.append(args.input)
    elif args.input_file:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
    else:
        # Default test texts
        texts = [
            "Hello, this is a test of the English DiffSinger model.",
            "I love singing and making music with artificial intelligence.",
            "The quick brown fox jumps over the lazy dog."
        ]
    
    # Run inference
    print(f"Running inference on {len(texts)} texts...")
    for i, text in enumerate(tqdm(texts)):
        # Generate filename
        filename = f"sample_{i:03d}"
        
        # Run inference
        output = model.inference(
            text,
            temperature=args.temperature
        )
        
        # Save mel spectrogram
        mel_pred = output['mel_pred']
        mel_path = os.path.join(args.output_dir, f"{filename}.png")
        plot_mel(mel_pred, mel_path)
        
        # Save waveform if available
        if output['wav_pred'] is not None:
            wav_path = os.path.join(args.output_dir, f"{filename}.wav")
            sf.write(wav_path, output['wav_pred'], config.audio_sample_rate)
        
        # Save text
        with open(os.path.join(args.output_dir, f"{filename}.txt"), 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"Generated sample {i}: {text}")
        print(f"  Mel: {mel_path}")
        if output['wav_pred'] is not None:
            print(f"  Wav: {wav_path}")
    
    print(f"Inference completed. Results saved in {args.output_dir}")

if __name__ == '__main__':
    main()
