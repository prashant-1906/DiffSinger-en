# English DiffSinger Adaptation

This is an adaptation of the [DiffSinger](https://github.com/MoonInTheRiver/DiffSinger) model for English language singing voice synthesis. The original DiffSinger model was designed primarily for Chinese language synthesis.

## Table of Contents
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Inference](#inference)
- [Model Architecture](#model-architecture)
- [File Structure](#file-structure)
- [Dependencies](#dependencies)
- [License](#license)

## Installation

1. Clone the DiffSinger repository:
```bash
git clone https://github.com/MoonInTheRiver/DiffSinger.git
cd DiffSinger
```

2. Install the dependencies:
```bash
pip install -r requirements.txt
```

3. Install additional dependencies for English text processing:
```bash
pip install g2p-en inflect nltk
```

4. Copy the English adaptation files to the repository:
```bash
# Place all the English adaptation files into their respective directories
```

## Data Preparation

### Dataset Format

For English datasets, prepare your data in the following format:

1. Create a text file with pairs of audio paths and their transcriptions:
```
/path/to/audio1.wav|This is the first sentence.
/path/to/audio2.wav|This is the second sentence.
```

2. Organize your dataset in the following structure:
```
data/raw/english_corpus/
├── train.txt
├── wavs/
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
```

### Processing the Dataset

Process your dataset using the provided preprocessing script:

```bash
python english_dataset_processor.py --config base
```

This will create the processed dataset in `data/processed/english_diffsinger/`.

## Training

To train the English DiffSinger model:

```bash
python english_training_script.py --config base
```

Available configurations:
- `base`: Standard model configuration
- `small`: Smaller, faster model with reduced parameters
- `large`: Larger model with more parameters for potentially better quality

Training parameters can be modified in the `english_dataset_config.py` file.

## Inference

After training, you can generate singing voice using:

```bash
python english_inference_script.py --checkpoint /path/to/checkpoint.pt --input "Text to synthesize"
```

Options:
- `--checkpoint`: Path to the trained model checkpoint
- `--input`: Text to synthesize
- `--input_file`: Path to a file containing multiple lines of text to synthesize
- `--output_dir`: Directory to save the generated mel spectrograms and audio files
- `--temperature`: Sampling temperature (default: 1.0)
- `--seed`: Random seed for reproducibility (default: 1234)

## Model Architecture

The English adaptation uses the same diffusion-based architecture as the original DiffSinger model, with the following modifications:

1. Custom English text frontend for phoneme conversion
2. English phoneme dictionary
3. Dataset processing specifically designed for English corpus

## File Structure

```
english_adaptation/
├── modules/
│   └── english_frontend.py        # English text processing module
├── english_phoneme_dict.py        # English phoneme dictionary
├── english_dataset_config.py      # Configuration for English dataset
├── english_dataset_processor.py   # Dataset preprocessing script
├── english_dataset_class.py       # PyTorch dataset class for English corpus
├── english_diffsinger_task.py     # Task definition for English DiffSinger
├── english_training_script.py     # Training script
├── english_inference_script.py    # Inference script
└── README.md                      # Documentation
```

## Dependencies

- Python 3.7+
- PyTorch 1.7+
- librosa
- g2p-en
- inflect
- nltk
- matplotlib
- numpy
- soundfile
- tqdm

## License

This adaptation follows the same license as the original DiffSinger project. Please refer to the DiffSinger repository for license details.

## Acknowledgements

- Original DiffSinger by [MoonInTheRiver](https://github.com/MoonInTheRiver/DiffSinger)
- Thanks to all the open-source contributors to the TTS community
