"""
English phoneme dictionary for DiffSinger
"""

# Silence and blank token
PAD = '<PAD>'
EOS = '<EOS>'
MASK = '<MASK>'
SP = 'SP'  # Short pause
AP = 'AP'  # Aspirated pause (breath)
SIL = 'SIL'  # Silence

# English vowels
VOWELS = [
    'aa',  # as in "father"
    'ae',  # as in "cat"
    'ah',  # as in "cut"
    'ao',  # as in "dog"
    'aw',  # as in "cow"
    'ay',  # as in "hide"
    'eh',  # as in "red"
    'er',  # as in "bird"
    'ey',  # as in "say"
    'ih',  # as in "sit"
    'iy',  # as in "see"
    'ow',  # as in "go"
    'oy',  # as in "boy"
    'uh',  # as in "could"
    'uw',  # as in "blue"
]

# English consonants
CONSONANTS = [
    'b',   # as in "bat"
    'ch',  # as in "chair"
    'd',   # as in "day"
    'dh',  # as in "this"
    'f',   # as in "fox"
    'g',   # as in "go"
    'hh',  # as in "hat"
    'jh',  # as in "just"
    'k',   # as in "cat"
    'l',   # as in "like"
    'm',   # as in "me"
    'n',   # as in "no"
    'ng',  # as in "sing"
    'p',   # as in "pen"
    'r',   # as in "red"
    's',   # as in "see"
    'sh',  # as in "she"
    't',   # as in "tea"
    'th',  # as in "thing"
    'v',   # as in "very"
    'w',   # as in "we"
    'y',   # as in "yes"
    'z',   # as in "zoo"
    'zh',  # as in "vision"
]

# Create the full phoneme set
PHONEME_SET = [PAD, EOS, MASK, SP, AP, SIL] + VOWELS + CONSONANTS

# Create a phoneme to id mapping
phoneme_to_id = {p: i for i, p in enumerate(PHONEME_SET)}
id_to_phoneme = {i: p for i, p in enumerate(PHONEME_SET)}

def get_phoneme_id(phoneme):
    """Get ID for a phoneme"""
    return phoneme_to_id.get(phoneme, phoneme_to_id[PAD])

def get_phoneme_from_id(idx):
    """Get phoneme from ID"""
    return id_to_phoneme.get(idx, PAD)

# Export the dictionary
phoneme_dict = {
    'pad': PAD,
    'eos': EOS,
    'mask': MASK,
    'sp': SP,
    'ap': AP,
    'sil': SIL,
    'vowels': VOWELS,
    'consonants': CONSONANTS,
    'phoneme_set': PHONEME_SET,
    'phoneme_to_id': phoneme_to_id,
    'id_to_phoneme': id_to_phoneme,
}
