import string
import emoji

MAIN_SPECIAL_CHARACTERS = string.punctuation + string.digits \
                          + string.whitespace
OTHER_SPECIAL_CHARACTERS = (
    "    　    ￼’“”–ー一▬…✦�­£​•€«»°·═"
    "×士＾˘⇓↓↑←→（）§″′´¿−±∈﻿¢ø‚„½¼¾¹²³―⁃，ˌ¸‹›ʺˈʻ¦‐⠀‰‑≤≥‖"
    "◆●■►▼▲▴∆▻¡★☆✱ːº。¯˜¥ɪ≈†上ン：∼⁄・♡✓⊕․．⋅÷１‟；،、¨ाাी्े◦˚"
    "゜ʼ≖ʼ¤ッツシ℃√！【】‿∞➤～πه۩☛₨➩☻๑٪♥ıॽ《‘©﴿٬？▷Г♫∟™ª₪®「—❖"
    "」﴾》"
)
EMOJI = list(emoji.EMOJI_DATA.keys())
SPECIAL_CHARACTERS = set(MAIN_SPECIAL_CHARACTERS + OTHER_SPECIAL_CHARACTERS)
SPECIAL_CHARACTERS.update(EMOJI)



class SpecialCharactersFilter:
    """Filter to keep samples with special-char ratio within a specific range."""

    def __init__(self, min_ratio=0.0, max_ratio=0.25):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def compute_stats(self, text_data):
        # Assuming SPECIAL_CHARACTERS, Fields, and StatsKeys are defined elsewhere
        sample = {'text': text_data, 'stats': {}}
        special_chars_count = sum(c in SPECIAL_CHARACTERS for c in text_data)
        sample['stats']['special_char_ratio'] = special_chars_count / len(text_data) if text_data else 0.0
        return sample

    def process(self, sample):
        # Retrieve the special_char_ratio from the sample using string keys
        special_char_ratio = sample.get('stats', {}).get('special_char_ratio', 0)
        # Check if the special_char_ratio is within the specified range
        return self.min_ratio <= special_char_ratio <= self.max_ratio



