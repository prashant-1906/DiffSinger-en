import re
import inflect
import nltk
from nltk.tokenize import word_tokenize
import g2p_en
from modules.commons.tts_modules import BaseTTSFrontend

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class EnglishFrontend(BaseTTSFrontend):
    def __init__(self, config=None):
        super().__init__(config)
        self.g2p = g2p_en.G2p()
        self.inflect = inflect.engine()
        self.punctuation = "!'(),.:;? "
        self.phoneme_map = self._get_phoneme_map()
        
    def _get_phoneme_map(self):
        """Define mapping from G2P phonemes to DiffSinger phoneme set"""
        # This is a basic mapping that can be expanded or modified
        phoneme_map = {
            # Vowels
            'AA': 'aa', 'AE': 'ae', 'AH': 'ah', 'AO': 'ao', 'AW': 'aw',
            'AY': 'ay', 'EH': 'eh', 'ER': 'er', 'EY': 'ey', 'IH': 'ih',
            'IY': 'iy', 'OW': 'ow', 'OY': 'oy', 'UH': 'uh', 'UW': 'uw',
            # Consonants
            'B': 'b', 'CH': 'ch', 'D': 'd', 'DH': 'dh', 'F': 'f',
            'G': 'g', 'HH': 'hh', 'JH': 'jh', 'K': 'k', 'L': 'l',
            'M': 'm', 'N': 'n', 'NG': 'ng', 'P': 'p', 'R': 'r',
            'S': 's', 'SH': 'sh', 'T': 't', 'TH': 'th', 'V': 'v',
            'W': 'w', 'Y': 'y', 'Z': 'z', 'ZH': 'zh'
        }
        return phoneme_map
    
    def preprocess_text(self, text):
        """Preprocess text by normalizing numbers and special characters"""
        # Lowercase everything
        text = text.lower()
        
        # Replace numbers with word form
        def _replace_number(match):
            number = match.group(0)
            return self.inflect.number_to_words(number)
            
        text = re.sub(r'\d+', _replace_number, text)
        
        # Replace special characters
        text = re.sub(r'[^a-zA-Z\s{}]'.format(re.escape(self.punctuation)), '', text)
        
        return text
    
    def get_phoneme_sequence(self, text):
        """Convert text to phoneme sequence"""
        # Preprocess the text
        text = self.preprocess_text(text)
        
        # Tokenize
        words = word_tokenize(text)
        
        # Convert words to phonemes
        all_phonemes = []
        for word in words:
            if word in self.punctuation:
                all_phonemes.append(word)
            else:
                phonemes = self.g2p(word)
                # Map to our phoneme set
                mapped_phonemes = []
                for p in phonemes:
                    # Remove stress markers (numbers)
                    if p[-1].isdigit():
                        p = p[:-1]
                    # Map to our phoneme set
                    if p in self.phoneme_map:
                        mapped_phonemes.append(self.phoneme_map[p])
                    else:
                        mapped_phonemes.append(p)
                all_phonemes.extend(mapped_phonemes)
                
        # Join with spaces
        return ' '.join(all_phonemes)
    
    def text_to_sequence(self, text, stage=None):
        """Convert text to sequence of phoneme IDs"""
        phonemes = self.get_phoneme_sequence(text).split()
        # Convert phonemes to IDs based on your phoneme dictionary
        # This requires integration with the phoneme dictionary in your config
        # For now, we'll return the phoneme list
        return phonemes

    def get_text(self, text, *args, **kwargs):
        """Main interface to convert text to features"""
        # Depending on your model requirements, add pitch/duration/alignment features
        phoneme_sequence = self.text_to_sequence(text)
        # Additional features should be added here
        return {'text': text, 'phonemes': phoneme_sequence}
