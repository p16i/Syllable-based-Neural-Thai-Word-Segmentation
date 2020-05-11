
from typing import Dict, List

# This dictionary is copied from https://github.com/rkcosmos/deepcut/blob/master/deepcut/utils.py#L6.
CHAR_TYPE = {
    u'กขฃคฆงจชซญฎฏฐฑฒณดตถทธนบปพฟภมยรลวศษสฬอ': 'c',
    u'ฅฉผฟฌหฮ': 'n',
    u"าะ ำ ิ ี ึ ื ั ู ุ": "v", # า ะ ำ ิ ี ึ ื ั ู ุ
    u'เแโใไ': 'w',
    ' ่ ้ ๊ ๋': 't',
    u'์ๆฯ.': 's', # ์  ๆ ฯ
    u'0123456789๑๒๓๔๕๖๗๘๙': 'd',
    u'"': 'q',
    u"‘": 'q',
    u"’": 'q',
    u"'": 'q',
    u' ': 'p',
    u'abcdefghijklmnopqrstuvwxyz': 's_e',
    u'ABCDEFGHIJKLMNOPQRSTUVWXYZ': 'b_e'
}

TAGS = [
    'c',
    'd',
    'n',
    'p',
    'q',
    's',
    't',
    'v',
    'w',
    's_e',
    'b_e',
    'o'
]

CHAR_TYPE_KEYS = sorted(CHAR_TYPE.values())

CHAR_TYPE_IX = dict(zip(TAGS, range(len(TAGS))))

CHAR_TO_CHARTYPE_IX = dict()

for k, v in CHAR_TYPE.items():
    for c in list(k):
        CHAR_TO_CHARTYPE_IX[c] = CHAR_TYPE_IX[v]

# manual adding token from our dict
CHAR_TO_CHARTYPE_IX["<PAD>"] = CHAR_TYPE_IX["p"]
CHAR_TO_CHARTYPE_IX["<PUNC>"] = CHAR_TYPE_IX["q"]
CHAR_TO_CHARTYPE_IX["<UNK>"] = CHAR_TYPE_IX["o"]

def get_char_type_ix(ch_seq: List[str]) -> List[int]:
    ix = []

    for c in ch_seq:
        ix.append(
            CHAR_TO_CHARTYPE_IX.get(c, CHAR_TYPE_IX["o"])
        )

    return ix

def get_total_char_types():
    return len(CHAR_TYPE_IX)

def get_char_type_cat(ch_type_ix):
    return list(map(lambda x: TAGS[x], ch_type_ix))