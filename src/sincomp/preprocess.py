# -*- coding: utf-8 -*-

"""
预处理方言读音数据的功能函数

基于正则表达式或 CRF 模型切分汉语音节声母、韵母、声调。
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import logging
import collections
import re
import numpy
import pandas
try:
    from sklearn_crfsuite import CRF
except ImportError:
    logging.warning('''sklearn_crfsuite not found, CRF based parser is disabled. Run following command to install sklearn_crfsuite:
        pip install sklearn-crfsuite''')


# 把读音中不规范的字符映射成规范 IPA 字符的映射表
_CHAR_MAP = {
    0x0030: 0x2205, # DIGIT ZERO -> EMPTY SET
    0x003a: 0x02d0, # COLON -> MODIFIER LETTER TRIANGULAR COLON
    0x0041: 0x1d00, # LATIN CAPITAL LETTER A -> LATIN LETTER SMALL CAPITAL A
    0x0042: 0x0062, # LATIN CAPITAL LETTER B -> LATIN SMALL LETTER B
    0x0043: 0x0063, # LATIN CAPITAL LETTER C -> LATIN SMALL LETTER C
    0x0044: 0x0064, # LATIN CAPITAL LETTER D -> LATIN SMALL LETTER D
    0x0045: 0x1d07, # LATIN CAPITAL LETTER E -> LATIN LETTER SMALL CAPITAL E
    0x0046: 0x0066, # LATIN CAPITAL LETTER F -> LATIN SMALL LETTER F
    0x0047: 0x0067, # LATIN CAPITAL LETTER G -> LATIN SMALL LETTER G
    0x0048: 0x0068, # LATIN CAPITAL LETTER H -> LATIN SMALL LETTER H
    0x0049: 0x026a, # LATIN CAPITAL LETTER I -> LATIN LETTER SMALL CAPITAL I
    0x004a: 0x006a, # LATIN CAPITAL LETTER J -> LATIN SMALL LETTER J
    0x004b: 0x006b, # LATIN CAPITAL LETTER K -> LATIN SMALL LETTER K
    0x004c: 0x006c, # LATIN CAPITAL LETTER L -> LATIN SMALL LETTER L
    0x004d: 0x006d, # LATIN CAPITAL LETTER M -> LATIN SMALL LETTER M
    0x004e: 0x006e, # LATIN CAPITAL LETTER N -> LATIN SMALL LETTER N
    0x004f: 0x006f, # LATIN CAPITAL LETTER O -> LATIN SMALL LETTER O
    0x0050: 0x0070, # LATIN CAPITAL LETTER P -> LATIN SMALL LETTER P
    0x0051: 0x0071, # LATIN CAPITAL LETTER Q -> LATIN SMALL LETTER Q
    0x0052: 0x0072, # LATIN CAPITAL LETTER R -> LATIN SMALL LETTER R
    0x0053: 0x0073, # LATIN CAPITAL LETTER S -> LATIN SMALL LETTER S
    0x0054: 0x0074, # LATIN CAPITAL LETTER T -> LATIN SMALL LETTER T
    0x0055: 0x0075, # LATIN CAPITAL LETTER U -> LATIN SMALL LETTER U
    0x0056: 0x0076, # LATIN CAPITAL LETTER V -> LATIN SMALL LETTER V
    0x0057: 0x0077, # LATIN CAPITAL LETTER W -> LATIN SMALL LETTER W
    0x0058: 0x0078, # LATIN CAPITAL LETTER X -> LATIN SMALL LETTER X
    0x0059: 0x028f, # LATIN CAPITAL LETTER Y -> LATIN LETTER SMALL CAPITAL Y
    0x005a: 0x007a, # LATIN CAPITAL LETTER Z -> LATIN SMALL LETTER Z
    0x0067: 0x0261, # LATIN SMALL LETTER G -> LATIN SMALL LETTER SCRIPT G
    0x007e: 0x0303, # TILDE -> COMBINING TILDE
    0x00b7: 0x2205, # MIDDLE DOT -> EMPTY SET
    0x00d8: 0x00f8, # LATIN CAPITAL LETTER O WITH STROKE -> LATIN SMALL LETTER O WITH STROKE
    0x0131: 0x0069, # LATIN SMALL LETTER DOTLESS I -> LATIN SMALL LETTER I
    0x019e: 0x014b, # LATIN SMALL LETTER N WITH LONG RIGHT LEG -> LATIN SMALL LETTER ENG
    0x01b2: 0x028b, # LATIN CAPITAL LETTER V WITH HOOK -> LATIN SMALL LETTER V WITH HOOK
    0x01dd: 0x0259, # LATIN SMALL LETTER TURNED E -> LATIN SMALL LETTER SCHWA
    0x01fe: 0x00f8, # LATIN CAPITAL LETTER O WITH STROKE AND ACUTE -> LATIN SMALL LETTER O WITH STROKE
    0x01ff: 0x00f8, # LATIN SMALL LETTER O WITH STROKE AND ACUTE -> LATIN SMALL LETTER O WITH STROKE
    0x0241: 0x0294, # LATIN CAPITAL LETTER GLOTTAL STOP -> LATIN LETTER GLOTTAL STOP
    0x02bb: 0x02b0, # MODIFIER LETTER TURNED COMMA -> MODIFIER LETTER SMALL H
    0x02dc: 0x0303, # SMALL TILDE -> COMBINING TILDE
    0x0307: 0x0329, # COMBINING DOT ABOVE -> COMBINING VERTICAL LINE BELOW
    0x030d: 0x0329, # COMBINING VERTICAL LINE ABOVE -> COMBINING VERTICAL LINE BELOW
    0x0311: 0x032f, # COMBINING INVERTED BREVE -> COMBINING INVERTED BREVE BELOW
    0x0323: 0x0329, # COMBINING DOT BELOW -> COMBINING VERTICAL LINE BELOW
    0x0331: 0x0320, # COMBINING MACRON BELOW -> COMBINING MINUS SIGN BELOW
    0x034a: 0x0303, # COMBINING NOT TILDE ABOVE -> COMBINING TILDE
    0x03b5: 0x025b, # GREEK SMALL LETTER EPSILON -> LATIN SMALL LETTER OPEN E
    0x1d02: 0x00e6, # LATIN SMALL LETTER TURNED AE -> LATIN SMALL LETTER AE
    0x2018: 0x02b0, # LEFT SINGLE QUOTATION MARK -> MODIFIER LETTER SMALL H
    0x25cb: 0x2205, # WHITE CIRCLE -> EMPTY SET
    0xff10: 0x2205, # FULLWIDTH DIGIT ZERO -> EMPTY SET
    0xff1a: 0x02d0, # FULLWIDTH COLON -> MODIFIER LETTER TRIANGULAR COLON
    0xff4b: 0x006b, # FULLWIDTH LATIN SMALL LETTER K -> LATIN SMALL LETTER K
    0xfffb: 0x2205, # INTERLINEAR ANNOTATION TERMINATOR -> EMPTY SET
}

_STRING_MAP = {
    '\u00f8\u0301': 'ø',
    '\u00e3': 'ã',
    '\u00f5': 'õ',
    '\u0115': 'ĕ',
    '\u0129': 'ĩ',
    '\u0169': 'ũ',
    '\u1e37': 'l̩',
    '\u1e43': 'm̩',
    '\u1ebd': 'ẽ',
    '\u1ef9': 'ỹ',
    '\u01eb': 'o̜',
    '\u1e47': 'n̩',
}

# IPA 字符集，包括一些现行 IPA 不收录但中国方言学界常用的字符
_CONSONANTS = {
    'b',        # U+0062, LATIN SMALL LETTER B
    'c',        # U+0063, LATIN SMALL LETTER C
    'd',        # U+0064, LATIN SMALL LETTER D
    'f',        # U+0066, LATIN SMALL LETTER F
    'h',        # U+0068, LATIN SMALL LETTER H
    'j',        # U+006A, LATIN SMALL LETTER J
    'k',        # U+006B, LATIN SMALL LETTER K
    'l',        # U+006C, LATIN SMALL LETTER L
    'm',        # U+006D, LATIN SMALL LETTER M
    'n',        # U+006E, LATIN SMALL LETTER N
    'p',        # U+0070, LATIN SMALL LETTER P
    'q',        # U+0071, LATIN SMALL LETTER Q
    'r',        # U+0072, LATIN SMALL LETTER R
    's',        # U+0073, LATIN SMALL LETTER S
    't',        # U+0074, LATIN SMALL LETTER T
    'v',        # U+0076, LATIN SMALL LETTER V
    'w',        # U+0077, LATIN SMALL LETTER W
    'x',        # U+0078, LATIN SMALL LETTER X
    'z',        # U+007A, LATIN SMALL LETTER Z
    'ç',        # U+00E7, LATIN SMALL LETTER C WITH CEDILLA
    'ð',        # U+00F0, LATIN SMALL LETTER ETH
    'ħ',        # U+0127, LATIN SMALL LETTER H WITH STROKE
    'ŋ',        # U+014B, LATIN SMALL LETTER ENG
    'ƈ',        # U+0188, LATIN SMALL LETTER C WITH HOOK
    'ƙ',        # U+0199, LATIN SMALL LETTER K WITH HOOK
    'ƛ',        # U+019B, LATIN SMALL LETTER LAMBDA WITH STROKE
    'ƥ',        # U+01A5, LATIN SMALL LETTER P WITH HOOK
    'ƭ',        # U+01AD, LATIN SMALL LETTER T WITH HOOK
    'ǀ',        # U+01C0, LATIN LETTER DENTAL CLICK
    'ǁ',        # U+01C1, LATIN LETTER LATERAL CLICK
    'ǂ',        # U+01C2, LATIN LETTER ALVEOLAR CLICK
    'ǃ',        # U+01C3, LATIN LETTER RETROFLEX CLICK
    'ȡ',        # U+0221, LATIN SMALL LETTER D WITH CURL
    'ȴ',        # U+0234, LATIN SMALL LETTER L WITH CURL
    'ȵ',        # U+0235, LATIN SMALL LETTER N WITH CURL
    'ȶ',        # U+0236, LATIN SMALL LETTER T WITH CURL
    'ɓ',        # U+0253, LATIN SMALL LETTER B WITH HOOK
    'ɕ',        # U+0255, LATIN SMALL LETTER C WITH CURL
    'ɖ',        # U+0256, LATIN SMALL LETTER D WITH TAIL
    'ɗ',        # U+0257, LATIN SMALL LETTER D WITH HOOK
    'ɟ',        # U+025F, LATIN SMALL LETTER DOTLESS J WITH STROKE
    'ɠ',        # U+0260, LATIN SMALL LETTER G WITH HOOK
    'ɡ',        # U+0261, LATIN SMALL LETTER SCRIPT G
    'ɢ',        # U+0262, LATIN LETTER SMALL CAPITAL G
    'ɣ',        # U+0263, LATIN SMALL LETTER GAMMA
    'ɥ',        # U+0265, LATIN SMALL LETTER TURNED H
    'ɦ',        # U+0266, LATIN SMALL LETTER H WITH HOOK
    'ɧ',        # U+0267, LATIN SMALL LETTER HENG WITH HOOK
    'ɫ',        # U+026B, LATIN SMALL LETTER L WITH MIDDLE TILDE
    'ɬ',        # U+026C, LATIN SMALL LETTER L WITH BELT
    'ɭ',        # U+026D, LATIN SMALL LETTER L WITH RETROFLEX HOOK
    'ɮ',        # U+026E, LATIN SMALL LETTER LEZH
    'ɰ',        # U+0270, LATIN SMALL LETTER TURNED M WITH LONG LEG
    'ɱ',        # U+0271, LATIN SMALL LETTER M WITH HOOK
    'ɲ',        # U+0272, LATIN SMALL LETTER N WITH LEFT HOOK
    'ɳ',        # U+0273, LATIN SMALL LETTER N WITH RETROFLEX HOOK
    'ɴ',        # U+0274, LATIN LETTER SMALL CAPITAL N
    'ɸ',        # U+0278, LATIN SMALL LETTER PHI
    'ɹ',        # U+0279, LATIN SMALL LETTER TURNED R
    'ɺ',        # U+027A, LATIN SMALL LETTER TURNED R WITH LONG LEG
    'ɻ',        # U+027B, LATIN SMALL LETTER TURNED R WITH HOOK
    'ɽ',        # U+027D, LATIN SMALL LETTER R WITH TAIL
    'ɾ',        # U+027E, LATIN SMALL LETTER R WITH FISHHOOK
    'ʀ',        # U+0280, LATIN LETTER SMALL CAPITAL R
    'ʁ',        # U+0281, LATIN LETTER SMALL CAPITAL INVERTED R
    'ʂ',        # U+0282, LATIN SMALL LETTER S WITH HOOK
    'ʃ',        # U+0283, LATIN SMALL LETTER ESH
    'ʄ',        # U+0284, LATIN SMALL LETTER DOTLESS J WITH STROKE AND HOOK
    'ʇ',        # U+0287, LATIN SMALL LETTER TURNED T
    'ʈ',        # U+0288, LATIN SMALL LETTER T WITH RETROFLEX HOOK
    'ʋ',        # U+028B, LATIN SMALL LETTER V WITH HOOK
    'ʍ',        # U+028D, LATIN SMALL LETTER TURNED W
    'ʎ',        # U+028E, LATIN SMALL LETTER TURNED Y
    'ʐ',        # U+0290, LATIN SMALL LETTER Z WITH RETROFLEX HOOK
    'ʑ',        # U+0291, LATIN SMALL LETTER Z WITH CURL
    'ʒ',        # U+0292, LATIN SMALL LETTER EZH
    'ʔ',        # U+0294, LATIN LETTER GLOTTAL STOP
    'ʕ',        # U+0295, LATIN LETTER PHARYNGEAL VOICED FRICATIVE
    'ʖ',        # U+0296, LATIN LETTER INVERTED GLOTTAL STOP
    'ʗ',        # U+0297, LATIN LETTER STRETCHED C
    'ʘ',        # U+0298, LATIN LETTER BILABIAL CLICK
    'ʙ',        # U+0299, LATIN LETTER SMALL CAPITAL B
    'ʛ',        # U+029B, LATIN LETTER SMALL CAPITAL G WITH HOOK
    'ʜ',        # U+029C, LATIN LETTER SMALL CAPITAL H
    'ʝ',        # U+029D, LATIN SMALL LETTER J WITH CROSSED-TAIL
    'ʞ',        # U+029E, LATIN SMALL LETTER TURNED K
    'ʟ',        # U+029F, LATIN LETTER SMALL CAPITAL L
    'ʠ',        # U+02A0, LATIN SMALL LETTER Q WITH HOOK
    'ʡ',        # U+02A1, LATIN LETTER GLOTTAL STOP WITH STROKE
    'ʢ',        # U+02A2, LATIN LETTER REVERSED GLOTTAL STOP WITH STROKE
    'ʣ',        # U+02A3, LATIN SMALL LETTER DZ DIGRAPH
    'ʤ',        # U+02A4, LATIN SMALL LETTER DEZH DIGRAPH
    'ʥ',        # U+02A5, LATIN SMALL LETTER DZ DIGRAPH WITH CURL
    'ʦ',        # U+02A6, LATIN SMALL LETTER TS DIGRAPH
    'ʧ',        # U+02A7, LATIN SMALL LETTER TESH DIGRAPH
    'ʨ',        # U+02A8, LATIN SMALL LETTER TC DIGRAPH WITH CURL
    'Φ',        # U+03A6, GREEK CAPITAL LETTER PHI
    'β',        # U+03B2, GREEK SMALL LETTER BETA
    'θ',        # U+03B8, GREEK SMALL LETTER THETA
    'λ',        # U+03BB, GREEK SMALL LETTER LAMDA
    'χ',        # U+03C7, GREEK SMALL LETTER CHI
    'ᶑ',        # U+1D91, LATIN SMALL LETTER D WITH HOOK AND TAIL
    '‼',        # U+203C, DOUBLE EXCLAMATION MARK
    'ⱱ',        # U+2C71, LATIN SMALL LETTER V WITH RIGHT HOOK
    'ⱳ',        # U+2C73, LATIN SMALL LETTER W WITH HOOK
    'ꞎ',        # U+A78E, LATIN SMALL LETTER L WITH RETROFLEX HOOK AND BELT
}

_VOWELS = {
    'a',        # U+0061, LATIN SMALL LETTER A
    'e',        # U+0065, LATIN SMALL LETTER E
    'i',        # U+0069, LATIN SMALL LETTER I
    'o',        # U+006F, LATIN SMALL LETTER O
    'u',        # U+0075, LATIN SMALL LETTER U
    'y',        # U+0079, LATIN SMALL LETTER Y
    'æ',        # U+00E6, LATIN SMALL LETTER AE
    'ø',        # U+00F8, LATIN SMALL LETTER O WITH STROKE
    'ü',        # U+00FC, LATIN SMALL LETTER U WITH DIAERESIS
    'œ',        # U+0153, LATIN SMALL LIGATURE OE
    'ɐ',        # U+0250, LATIN SMALL LETTER TURNED A
    'ɑ',        # U+0251, LATIN SMALL LETTER ALPHA
    'ɒ',        # U+0252, LATIN SMALL LETTER TURNED ALPHA
    'ɔ',        # U+0254, LATIN SMALL LETTER OPEN O
    'ɘ',        # U+0258, LATIN SMALL LETTER REVERSED E
    'ə',        # U+0259, LATIN SMALL LETTER SCHWA
    'ɚ',        # U+025A, LATIN SMALL LETTER SCHWA WITH HOOK
    'ɛ',        # U+025B, LATIN SMALL LETTER OPEN E
    'ɜ',        # U+025C, LATIN SMALL LETTER REVERSED OPEN E
    'ɝ',        # U+025D, LATIN SMALL LETTER REVERSED OPEN E WITH HOOK
    'ɞ',        # U+025E, LATIN SMALL LETTER CLOSED REVERSED OPEN E
    'ɤ',        # U+0264, LATIN SMALL LETTER RAMS HORN
    'ɨ',        # U+0268, LATIN SMALL LETTER I WITH STROKE
    'ɪ',        # U+026A, LATIN LETTER SMALL CAPITAL I
    'ɯ',        # U+026F, LATIN SMALL LETTER TURNED M
    'ɵ',        # U+0275, LATIN SMALL LETTER BARRED O
    'ɶ',        # U+0276, LATIN LETTER SMALL CAPITAL OE
    'ɷ',        # U+0277, LATIN SMALL LETTER CLOSED OMEGA
    'ɿ',        # U+027F, LATIN SMALL LETTER REVERSED R WITH FISHHOOK
    'ʅ',        # U+0285, LATIN SMALL LETTER SQUAT REVERSED ESH
    'ʉ',        # U+0289, LATIN SMALL LETTER U BAR
    'ʊ',        # U+028A, LATIN SMALL LETTER UPSILON
    'ʌ',        # U+028C, LATIN SMALL LETTER TURNED V
    'ʏ',        # U+028F, LATIN LETTER SMALL CAPITAL Y
    'ʮ',        # U+02AE, LATIN SMALL LETTER TURNED H WITH FISHHOOK
    'ʯ',        # U+02AF, LATIN SMALL LETTER TURNED H WITH FISHHOOK AND TAIL
    'ω',        # U+03C9, GREEK SMALL LETTER OMEGA
    'ᴀ',        # U+1D00, LATIN LETTER SMALL CAPITAL A
    'ᴇ',        # U+1D07, LATIN LETTER SMALL CAPITAL E
    'ᵻ',        # U+1D7B, LATIN SMALL CAPITAL LETTER I WITH STROKE
    'ᵿ',        # U+1D7F, LATIN SMALL LETTER UPSILON WITH STROKE
}

_DIACRITICS = {
    'ʰ',        # U+02B0, MODIFIER LETTER SMALL H
    'ʱ',        # U+02B1, MODIFIER LETTER SMALL H WITH HOOK
    'ʲ',        # U+02B2, MODIFIER LETTER SMALL J
    'ʳ',        # U+02B3, MODIFIER LETTER SMALL R
    'ʴ',        # U+02B4, MODIFIER LETTER SMALL TURNED R
    'ʵ',        # U+02B5, MODIFIER LETTER SMALL TURNED R WITH HOOK
    'ʶ',        # U+02B6, MODIFIER LETTER SMALL CAPITAL INVERTED R
    'ʷ',        # U+02B7, MODIFIER LETTER SMALL W
    'ʼ',        # U+02BC, MODIFIER LETTER APOSTROPHE
    'ˀ',        # U+02C0, MODIFIER LETTER GLOTTAL STOP
    '˔',        # U+02D4, MODIFIER LETTER UP TACK
    '˕',        # U+02D5, MODIFIER LETTER DOWN TACK
    '˚',        # U+02DA, RING ABOVE
    '˞',        # U+02DE, MODIFIER LETTER RHOTIC HOOK
    'ˠ',        # U+02E0, MODIFIER LETTER SMALL GAMMA
    'ˡ',        # U+02E1, MODIFIER LETTER SMALL L
    'ˣ',        # U+02E3, MODIFIER LETTER SMALL X
    'ˤ',        # U+02E4, MODIFIER LETTER SMALL REVERSED GLOTTAL STOP
    '˺',        # U+02FA, MODIFIER LETTER END HIGH TONE
    '̀',        # U+0300, COMBINING GRAVE ACCENT
    '́',        # U+0301, COMBINING ACUTE ACCENT
    '̂',        # U+0302, COMBINING CIRCUMFLEX ACCENT
    '̃',        # U+0303, COMBINING TILDE
    '̄',        # U+0304, COMBINING MACRON
    '̈',        # U+0308, COMBINING DIAERESIS
    '̊',        # U+030A, COMBINING RING ABOVE
    '̋',        # U+030B, COMBINING DOUBLE ACUTE ACCENT
    '̌',        # U+030C, COMBINING CARON
    '̏',        # U+030F, COMBINING DOUBLE GRAVE ACCENT
    '̘',        # U+0318, COMBINING LEFT TACK BELOW
    '̙',        # U+0319, COMBINING RIGHT TACK BELOW
    '̚',        # U+031A, COMBINING LEFT ANGLE ABOVE
    '̜',        # U+031C, COMBINING LEFT HALF RING BELOW
    '̝',        # U+031D, COMBINING UP TACK BELOW
    '̞',        # U+031E, COMBINING DOWN TACK BELOW
    '̟',        # U+031F, COMBINING PLUS SIGN BELOW
    '̠',        # U+0320, COMBINING MINUS SIGN BELOW
    '̢',        # U+0322, COMBINING RETROFLEX HOOK BELOW
    '̤',        # U+0324, COMBINING DIAERESIS BELOW
    '̥',        # U+0325, COMBINING RING BELOW
    '̩',        # U+0329, COMBINING VERTICAL LINE BELOW
    '̪',        # U+032A, COMBINING BRIDGE BELOW
    '̫',        # U+032B, COMBINING INVERTED DOUBLE ARCH BELOW
    '̬',        # U+032C, COMBINING CARON BELOW
    '̮',        # U+032E, COMBINING BREVE BELOW
    '̯',        # U+032F, COMBINING INVERTED BREVE BELOW
    '̰',        # U+0330, COMBINING TILDE BELOW
    '̳',        # U+0333, COMBINING DOUBLE LOW LINE
    '̴',        # U+0334, COMBINING TILDE OVERLAY
    '̵',        # U+0335, COMBINING SHORT STROKE OVERLAY
    '̹',        # U+0339, COMBINING RIGHT HALF RING BELOW
    '̺',        # U+033A, COMBINING INVERTED BRIDGE BELOW
    '̻',        # U+033B, COMBINING SQUARE BELOW
    '̼',        # U+033C, COMBINING SEAGULL BELOW
    '̽',        # U+033D, COMBINING X ABOVE
    '͇',        # U+0347, COMBINING EQUALS SIGN BELOW
    '͜',        # U+035C, COMBINING DOUBLE BREVE BELOW
    '͡',        # U+0361, COMBINING DOUBLE INVERTED BREVE
    'ᵃ',        # U+1D43, MODIFIER LETTER SMALL A
    'ᵄ',        # U+1D44, MODIFIER LETTER SMALL TURNED A
    'ᵅ',        # U+1D45, MODIFIER LETTER SMALL ALPHA
    'ᵇ',        # U+1D47, MODIFIER LETTER SMALL B
    'ᵈ',        # U+1D48, MODIFIER LETTER SMALL D
    'ᵉ',        # U+1D49, MODIFIER LETTER SMALL E
    'ᵊ',        # U+1D4A, MODIFIER LETTER SMALL SCHWA
    'ᵋ',        # U+1D4B, MODIFIER LETTER SMALL OPEN E
    'ᵏ',        # U+1D4F, MODIFIER LETTER SMALL K
    'ᵐ',        # U+1D50, MODIFIER LETTER SMALL M
    'ᵑ',        # U+1D51, MODIFIER LETTER SMALL ENG
    'ᵒ',        # U+1D52, MODIFIER LETTER SMALL O
    'ᵓ',        # U+1D53, MODIFIER LETTER SMALL OPEN O
    'ᵖ',        # U+1D56, MODIFIER LETTER SMALL P
    'ᵗ',        # U+1D57, MODIFIER LETTER SMALL T
    'ᵘ',        # U+1D58, MODIFIER LETTER SMALL U
    'ᵚ',        # U+1D5A, MODIFIER LETTER SMALL TURNED M
    'ᵛ',        # U+1D5B, MODIFIER LETTER SMALL V
    'ᵝ',        # U+1D5D, MODIFIER LETTER SMALL BETA
    'ᵡ',        # U+1D61, MODIFIER LETTER SMALL CHI
    'ᶛ',        # U+1D9B, MODIFIER LETTER SMALL TURNED ALPHA
    'ᶜ',        # U+1D9C, MODIFIER LETTER SMALL C
    'ᶝ',        # U+1D9D, MODIFIER LETTER SMALL C WITH CURL
    'ᶞ',        # U+1D9E, MODIFIER LETTER SMALL ETH
    'ᶟ',        # U+1D9F, MODIFIER LETTER SMALL REVERSED OPEN E
    'ᶠ',        # U+1DA0, MODIFIER LETTER SMALL F
    'ᶡ',        # U+1DA1, MODIFIER LETTER SMALL DOTLESS J WITH STROKE
    'ᶢ',        # U+1DA2, MODIFIER LETTER SMALL SCRIPT G
    'ᶣ',        # U+1DA3, MODIFIER LETTER SMALL TURNED H
    'ᶤ',        # U+1DA4, MODIFIER LETTER SMALL I WITH STROKE
    'ᶦ',        # U+1DA6, MODIFIER LETTER SMALL CAPITAL I
    'ᶨ',        # U+1DA8, MODIFIER LETTER SMALL J WITH CROSSED-TAIL
    'ᶩ',        # U+1DA9, MODIFIER LETTER SMALL L WITH RETROFLEX HOOK
    'ᶬ',        # U+1DAC, MODIFIER LETTER SMALL M WITH HOOK
    'ᶭ',        # U+1DAD, MODIFIER LETTER SMALL TURNED M WITH LONG LEG
    'ᶮ',        # U+1DAE, MODIFIER LETTER SMALL N WITH LEFT HOOK
    'ᶯ',        # U+1DAF, MODIFIER LETTER SMALL N WITH RETROFLEX HOOK
    'ᶰ',        # U+1DB0, MODIFIER LETTER SMALL CAPITAL N
    'ᶲ',        # U+1DB2, MODIFIER LETTER SMALL PHI
    'ᶳ',        # U+1DB3, MODIFIER LETTER SMALL S WITH HOOK
    'ᶴ',        # U+1DB4, MODIFIER LETTER SMALL ESH
    'ᶶ',        # U+1DB6, MODIFIER LETTER SMALL U BAR
    'ᶷ',        # U+1DB7, MODIFIER LETTER SMALL UPSILON
    'ᶹ',        # U+1DB9, MODIFIER LETTER SMALL V WITH HOOK
    'ᶺ',        # U+1DBA, MODIFIER LETTER SMALL TURNED V
    'ᶻ',        # U+1DBB, MODIFIER LETTER SMALL Z
    'ᶼ',        # U+1DBC, MODIFIER LETTER SMALL Z WITH RETROFLEX HOOK
    'ᶽ',        # U+1DBD, MODIFIER LETTER SMALL Z WITH CURL
    'ᶾ',        # U+1DBE, MODIFIER LETTER SMALL EZH
    'ᶿ',        # U+1DBF, MODIFIER LETTER SMALL THETA
    '᷄',        # U+1DC4, COMBINING MACRON-ACUTE
    '᷅',        # U+1DC5, COMBINING GRAVE-MACRON
    '᷆',        # U+1DC6, COMBINING MACRON-GRAVE
    '᷇',        # U+1DC7, COMBINING ACUTE-MACRON
    '᷈',        # U+1DC8, COMBINING GRAVE-ACUTE-GRAVE
    '᷉',        # U+1DC9, COMBINING ACUTE-GRAVE-ACUTE
    '\u2009',   # U+2009, THIN SPACE
    '’',        # U+2019, RIGHT SINGLE QUOTATION MARK
    'ⁱ',        # U+2071, SUPERSCRIPT LATIN SMALL LETTER I
    'ⁿ',        # U+207F, SUPERSCRIPT LATIN SMALL LETTER N
    '↓',        # U+2193, DOWNWARDS ARROW
}

_SUPRASEGMENTALS = {
    '.',        # U+002E, FULL STOP
    '|',        # U+007C, VERTICAL LINE
    'ˈ',        # U+02C8, MODIFIER LETTER VERTICAL LINE
    'ˌ',        # U+02CC, MODIFIER LETTER LOW VERTICAL LINE
    'ː',        # U+02D0, MODIFIER LETTER TRIANGULAR COLON
    'ˑ',        # U+02D1, MODIFIER LETTER HALF TRIANGULAR COLON
    '̆',        # U+0306, COMBINING BREVE
    '‿',        # U+203F, UNDERTIE
}

_TONES = {
    '1',        # U+0031, DIGIT ONE
    '2',        # U+0032, DIGIT TWO
    '3',        # U+0033, DIGIT THREE
    '4',        # U+0034, DIGIT FOUR
    '5',        # U+0035, DIGIT FIVE
    '²',        # U+00B2, SUPERSCRIPT TWO
    '³',        # U+00B3, SUPERSCRIPT THREE
    '¹',        # U+00B9, SUPERSCRIPT ONE
    '⁴',        # U+2074, SUPERSCRIPT FOUR
    '⁵',        # U+2075, SUPERSCRIPT FIVE
    '˥',        # U+02E5, MODIFIER LETTER EXTRA-HIGH TONE BAR
    '˦',        # U+02E6, MODIFIER LETTER HIGH TONE BAR
    '˧',        # U+02E7, MODIFIER LETTER MID TONE BAR
    '˨',        # U+02E8, MODIFIER LETTER LOW TONE BAR
    '˩',        # U+02E9, MODIFIER LETTER EXTRA-LOW TONE BAR
    '↗',        # U+2197, NORTH EAST ARROW
    '↘',        # U+2198, SOUTH EAST ARROW
    'ꜛ',        # U+A71B, MODIFIER LETTER RAISED UP ARROW
    'ꜜ',        # U+A71C, MODIFIER LETTER RAISED DOWN ARROW
}

_LETTERS = _CONSONANTS | _VOWELS
_IPA = _LETTERS | _DIACRITICS | _SUPRASEGMENTALS | _TONES

# IPA 字符对应的类型映射表
_TYPE_MAP = {}
for s, t in (
    (_CONSONANTS, 'consonant'),
    (_VOWELS, 'vowel'),
    (_DIACRITICS, 'diacritic'),
    (_SUPRASEGMENTALS, 'suprasegmental'),
    (_TONES, 'tone')
):
    _TYPE_MAP.update((c, t) for c in s)

# IPA 字符对应的发音方法映射表
_MANNER_MAP = {
    'b': 'plosive',
    'c': 'plosive',
    'd': 'plosive',
    'f': 'non-sibilant-fricative',
    'h': 'non-sibilant-fricative',
    'j': 'approximant',
    'k': 'plosive',
    'l': 'lateral-approximant',
    'm': 'nasal',
    'n': 'nasal',
    'p': 'plosive',
    'q': 'plosive',
    'r': 'trill',
    's': 'sibilant-fricative',
    't': 'plosive',
    'v': 'non-sibilant-fricative',
    'w': 'approximant',
    'x': 'non-sibilant-fricative',
    'z': 'sibilant-fricative',
    'ç': 'non-sibilant-fricative',
    'ð': 'non-sibilant-fricative',
    'ħ': 'non-sibilant-fricative',
    'ŋ': 'nasal',
    'ƈ': 'implosive',
    'ƙ': 'implosive',
    'ƛ': 'lateral-affricate',
    'ƥ': 'implosive',
    'ƭ': 'implosive',
    'ǀ': 'click',
    'ǁ': 'lateral-click',
    'ǂ': 'click',
    'ǃ': 'click',
    'ȴ': 'lateral-approximant',
    'ɓ': 'implosive',
    'ɕ': 'sibilant-fricative',
    'ɖ': 'plosive',
    'ɗ': 'implosive',
    'ɟ': 'plosive',
    'ɠ': 'implosive',
    'ɡ': 'plosive',
    'ɢ': 'plosive',
    'ɣ': 'non-sibilant-fricative',
    'ɥ': 'approximant',
    'ɦ': 'non-sibilant-fricative',
    'ɧ': 'sibilant-fricative',
    'ɫ': 'lateral-approximant',
    'ɬ': 'lateral-fricative',
    'ɭ': 'lateral-approximant',
    'ɮ': 'lateral-fricative',
    'ɰ': 'approximant',
    'ɱ': 'nasal',
    'ɲ': 'nasal',
    'ɳ': 'nasal',
    'ɴ': 'nasal',
    'ɸ': 'non-sibilant-fricative',
    'ɹ': 'approximant',
    'ɺ': 'lateral-flap',
    'ɻ': 'approximant',
    'ɽ': 'flap',
    'ɾ': 'flap',
    'ʀ': 'trill',
    'ʁ': 'non-sibilant-fricative',
    'ʂ': 'sibilant-fricative',
    'ʃ': 'sibilant-fricative',
    'ʄ': 'implosive',
    'ʇ': 'click',
    'ʈ': 'plosive',
    'ʋ': 'approximant',
    'ʍ': 'approximant',
    'ʎ': 'lateral-approximant',
    'ʐ': 'sibilant-fricative',
    'ʑ': 'sibilant-fricative',
    'ʒ': 'sibilant-fricative',
    'ʔ': 'plosive',
    'ʕ': 'non-sibilant-fricative',
    'ʖ': 'lateral-click',
    'ʗ': 'click',
    'ʘ': 'click',
    'ʙ': 'trill',
    'ʛ': 'implosive',
    'ʜ': 'trill',
    'ʝ': 'non-sibilant-fricative',
    'ʞ': 'click',
    'ʟ': 'lateral-approximant',
    'ʠ': 'implosive',
    'ʡ': 'plosive',
    'ʢ': 'trill',
    'ʣ': 'sibilant-affricate',
    'ʤ': 'sibilant-affricate',
    'ʥ': 'sibilant-affricate',
    'ʦ': 'sibilant-affricate',
    'ʧ': 'sibilant-affricate',
    'ʨ': 'sibilant-affricate',
    'Φ': 'non-sibilant-fricative',
    'β': 'non-sibilant-fricative',
    'θ': 'non-sibilant-fricative',
    'λ': 'lateral-affricate',
    'χ': 'non-sibilant-fricative',
    'ᶑ': 'implosive',
    '‼': 'click',
    'ⱱ': 'flap',
    'ⱳ': 'flap',
    'ꞎ': 'lateral-fricative',
}

# 声调从常规数字转换成上标
_TONE_TO_SUPERSCRIPT = {
    0x0031: 0x00b9, # DIGIT ONE -> SUPERSCRIPT ONE
    0x0032: 0x00b2, # DIGIT TWO -> SUPERSCRIPT TWO
    0x0033: 0x00b3, # DIGIT THREE -> SUPERSCRIPT THREE
    0x0034: 0x2074, # DIGIT FOUR -> SUPERSCRIPT FOUR
    0x0035: 0x2075, # DIGIT FIVE -> SUPERSCRIPT FIVE
}

# 声调从上标数字转成常规数字
_TONE_TO_NORMAL = {
    0x00b2: 0x0032, # SUPERSCRIPT TWO -> DIGIT TWO
    0x00b3: 0x0033, # SUPERSCRIPT THREE -> DIGIT THREE
    0x00b9: 0x0031, # SUPERSCRIPT ONE -> DIGIT ONE
    0x2074: 0x0034, # SUPERSCRIPT FOUR -> DIGIT FOUR
    0x2075: 0x0035, # SUPERSCRIPT FIVE -> DIGIT FIVE
}


def clean_ipa(raw: pandas.Series, force: bool = False) -> str:
    """
    清洗方言读音 IPA

    把读音中不规范的 IPA 字符映射成规范 IPA 字符。

    Parameters:
        raw: 包含原始 IPA 字符串的列表
        force: 为真时强制删除所有非 IPA 字符

    Returns:
        clean: 清洗后的 IPA 字符串
    """

    clean = raw.str.strip().str.translate(_CHAR_MAP)
    for k, v in _STRING_MAP.items():
        clean = clean.str.replace(k, v, regex=False)

    if force:
        clean = clean.str.replace(f'[^{"".join(_IPA)}]', '', regex=True)

    return clean

def clean_initial(raw: pandas.Series) -> pandas.Series:
    """
    清洗方言字音数据中的声母

    Parameters:
        raw: 方言字音声母列表

    Returns:
        clean: 清洗后的方言字音声母列表
    """

    # 允许单个空值符号作为声母，需特殊处理
    return raw.where(
        raw == '∅',
        raw.str.replace(
            f'[^{"".join(_CONSONANTS)}{"".join(_DIACRITICS)}]',
            '',
            regex=True
        )
    )

def clean_final(raw: pandas.Series) -> pandas.Series:
    """
    清洗方言字音数据中的韵母

    Parameters:
        raw: 方言字音韵母列表

    Returns:
        clean: 清洗后的方言字音韵母列表
    """

    return raw.str.replace(
        f'[^{"".join(_LETTERS)}{"".join(_DIACRITICS)}{"".join(_SUPRASEGMENTALS)}]',
        '',
        regex=True
    )

def clean_tone(raw: pandas.Series) -> pandas.Series:
    """
    清洗方言字音数据中的声调

    Parameters:
        raw: 方言字音声调列表

    Returns:
        clean: 清洗后的方言字音声调列表
    """

    # 部分数据集把轻声标为零声调
    return raw.where(
        raw == '∅',
        raw.str.replace(f'[^{"".join(_TONES)}]', '', regex=True)
    )

def normalize_initial(origin: pandas.Series) -> pandas.Series:
    """
    规范化方言字音数据中的声母

    Parameters:
        origin: 方言字音声母列表

    Returns:
        output: 规范化的方言字音声母列表
    """

    # 有些符号使用了多种写法，统一成较常用的一种
    return origin.str.translate({
        0x1d50: 0x006d, # MODIFIER LETTER SMALL M -> LATIN SMALL LETTER M
        0x1d51: 0x014b, # MODIFIER LETTER SMALL ENG -> LATIN SMALL LETTER ENG
        0x1d5b: 0x1db9, # MODIFIER LETTER SMALL V -> MODIFIER LETTER SMALL V WITH HOOK
        0x207f: 0x006e, # SUPERSCRIPT LATIN SMALL LETTER N -> LATIN SMALL LETTER N
    }) \
        .str.replace('\u02a3', 'dz', regex=False) \
        .str.replace('\u02a4', 'dʒ', regex=False) \
        .str.replace('\u02a5', 'dʑ', regex=False) \
        .str.replace('\u02a6', 'ts', regex=False) \
        .str.replace('\u02a7', 'tʃ', regex=False) \
        .str.replace('\u02a8', 'tɕ', regex=False) \
        .str.replace('([kɡŋhɦ].?)w', r'\1ʷ', regex=True) \
        .str.replace('([kɡŋhɦ].?)[vʋ]', r'\1ᶹ', regex=True) \
        .str.replace('([^ʔ∅])h', r'\1ʰ', regex=True) \
        .str.replace('([^ʔ∅])ɦ', r'\1ʱ', regex=True) \
        .str.replace('([bdɡvzʐʑʒɾ])ʱ', r'\1ʰ', regex=True) \
        .str.replace('([ʰʱ])([ʷᶹ])', r'\2\1', regex=True)

def tone2super(origin: pandas.Series) -> pandas.Series:
    """
    把字符串中的声调转成上标数字

    Parameters:
        origin: 包含声调数字的原始字符串列表

    Returns:
        output: 声调转成上标数字的结果字符串列表
    """

    return origin.str.translate(_TONE_TO_SUPERSCRIPT)


def transform(
    data: pandas.DataFrame,
    index: str = 'did',
    columns: str = 'cid',
    values: list[str] | None = None,
    aggfunc: str | collections.abc.Callable = 'first',
    **kwargs
) -> pandas.DataFrame:
    """
    把方言读音数据长表转换为宽表

    Parameters:
        data: 待转换的读音数据长表
        index: 指明以原始表的哪一列为行
        columns: 指明以原始表的哪一列为一级列
        values: 用于变换的列，变换后成为二级列，为空保留所有列
        aggfunc: 相同的行列有多个记录的，使用 aggfunc 函数合并
        kwargs: 透传给 pandas.DataFrame.pivot_table

    Returns:
        output: 转换格式得到的数据宽表
    """

    output = data.pivot_table(
        values,
        index=index,
        columns=columns,
        aggfunc=aggfunc,
        sort=False,
        **kwargs
    )

    # 如果列名为多层级，把指定的列名上移到最高层级
    if output.columns.nlevels > 1:
        output = output.swaplevel(axis=1).reindex(
            pandas.MultiIndex.from_product((
                output.columns.levels[1],
                output.columns.levels[0]
            )),
            axis=1
        )

    return output


def str2fea(s: str) -> dict[str, str]:
    """
    把字符串转化成序列标注模型需要的输入特征序列

    Parameters:
        s: 原始字符串

    Returns:
        features: 特征列表，每个元素是一个包含多组特征 key-value 对的字典
    """

    features = []

    for i in range(len(s)):
        fea = {}
        for p in range(-1, 2):
            pos = i + p
            if pos < 0:
                fea[f'{p:+d}:char'] = 'BOS'
            elif pos >= len(s):
                fea[f'{p:+d}:char'] = 'EOS'
            else:
                c = s[pos]
                fea.update({
                    f'{p:+d}:char': c,
                    f'{p:+d}:type': _TYPE_MAP.get(c, 'other')
                })

                try:
                    fea[f'{p:+d}:manner'] = _MANNER_MAP[c]
                except KeyError:
                    ...

        features.append(fea)

    return features

def segment(s: str, tags: list[str]) -> tuple[str | None, str | None, str | None]:
    """
    根据模型预测的标注序列切分音节

    Parameters:
        s: 待切分的音节字符串
        tags: 标注列表，长度和 `s` 相同

    Returns:
        initial, final, tone: 切分出的声母、韵母、声调字符串，如果切分失败均返回 None

    `tags` 必须遵循 BIOSE 标注形式，即每个标注最多由前后2部分组成，由横杠 - 分隔。
    前段表示某个元素的开始或结束：
        - B 表示 `s` 对应位置的字符为某个元素的开始
        - I 元素的中间
        - E 元素的结束
        - S 单个字符单独构成一个元素
        - O 不属于任何元素的其他字符，这种标签没有后段

    后段表示属于什么元素：
        - I 声母
        - F 韵母
        - T 声调
    """

    elements = {}

    for c, l in zip(s, tags):
        pos, _, e = l.partition('-')
        if pos == 'B' or pos == 'S':
            elements[e] = c
        elif pos == 'I' or pos == 'E':
            elements[e] = elements.get(e, '') + c

    return elements.get('I'), elements.get('F'), elements.get('T')


class RegexParser:
    """
    基于正则表达式切分方言音节声母、韵母、声调
    """

    def __init__(self, pattern: str):
        """
        Parameters:
            pattern: 用于切分音节的正则表达式，必须包含3个组，按顺序分别表示声母、韵母、声调
        """

        self.pattern = pattern

    def parse(self, syllable: str) -> tuple[str | None, str | None, str | None]:
        """
        切分单个音节

        Parameters:
            syllable: 待切分音节字符串

        Returns:
            initial, final, tone: 切分出的声母、韵母、声调字符串，如果切分失败均返回 None
        """

        match = re.search(self.pattern, syllable)
        return (None, None, None) if match is None else match.groups()

    def parse_batch(
        self,
        syllables: numpy.ndarray[str] | pandas.Series
    ) -> numpy.ndarray[str] | pandas.DataFrame:
        """
        切分批量音节

        Parameters:
            syllables: 待切分音节列表

        Returns:
            elements: 切分结果列表，行数和 `syllables` 相同，列依次为声母、韵母、声调
        """

        elements = pandas.Series(syllables).str.extract(self.pattern)
        if isinstance(syllables, pandas.Series):
            elements.columns = ['initial', 'final', 'tone']
        else:
            elements = elements.values

        return elements

    def __call__(
        self,
        input: str | numpy.ndarray[str] | pandas.Series
    ) -> tuple[str, str, str] | numpy.ndarray[str] | pandas.DataFrame:
        return self.parse(input) if isinstance(input, str) \
            else self.parse_batch(input)

class CRFParser:
    """
    基于 CRF 序列标注模型切分方言音节声母、韵母、声调
    """

    def __init__(self, path: str):
        """
        Parameters:
            path: 模型文件路径
        """

        self.model = CRF(model_filename=path)

    def parse(self, syllable: str) -> tuple[str | None, str | None, str | None]:
        """
        切分单个音节

        Parameters:
            syllable: 待切分音节字符串

        Returns:
            initial, final, tone: 切分出的声母、韵母、声调字符串，如果切分失败均返回 None
        """

        tags = self.model.predict_single(syllable)
        return segment(syllable, tags)

    def parse_batch(
        self,
        syllables: numpy.ndarray[str] | pandas.Series
    ) -> numpy.ndarray[str] | pandas.DataFrame:
        """
        切分批量音节

        Parameters:
            syllables: 待切分音节列表

        Returns:
            elements: 切分结果列表，行数和 `syllables` 相同，列依次为声母、韵母、声调
        """

        tags = self.model.predict(
            syllables.map(str2fea) if isinstance(syllables, pandas.Series) \
                else (str2fea(s) for s in syllables)
        )
        elements = numpy.asarray([segment(s, t) for s, t in zip(syllables, tags)])

        if isinstance(syllables, pandas.Series):
            elements = pandas.DataFrame(
                elements,
                index=syllables.index,
                columns=['initial', 'final', 'tone']
            )

        return elements

    def __call__(
        self,
        input: str | numpy.ndarray[str] | pandas.Series
    ) -> tuple[str, str, str] | numpy.ndarray[str] | pandas.DataFrame:
        return self.parse(input) if isinstance(input, str) else self.parse_batch(input)


# 默认的音节切分函数
parse = RegexParser(
    f'([{"".join(_DIACRITICS)}]*[{"".join(_CONSONANTS)}][{"".join(_CONSONANTS)}{"".join(_DIACRITICS)}]*|)'
    f'([{"".join(_DIACRITICS)}]*[{"".join(_LETTERS)}][{"".join(_LETTERS | _DIACRITICS | _SUPRASEGMENTALS)}]*)'
    f'([{"".join(_TONES)}]*)'
)
