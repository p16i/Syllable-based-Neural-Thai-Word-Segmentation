import numpy as np

def get_scheme(name):
    if name == "BI":
        return SchemeBI
    elif name == "SchemeA":
        return SchemeA
    elif name == "SchemeASyLevel":
        return SchemeASyLevel
    elif name == "SchemeB":
        return SchemeB
    elif name == "SchemeBSyLevel":
        return SchemeBSyLevel
    else:
        raise ValueError(f"No Scheme: {name} exists!!")


def find_word_boundaries(labels):
    labels = np.array(labels)
    b_loc = np.argwhere(labels == 1).reshape(-1).tolist()

    word_boundaries = list(zip(b_loc[:-1], b_loc[1:])) \
        + [(b_loc[-1], labels.shape[0])]

    return word_boundaries

class SchemeBI:
    
    num_tags = 2

    @staticmethod
    def encode(labels, sy_ix):
        """ don't nothing here"""

        return np.array(labels).astype(int)

    @staticmethod
    def decode(pred_probs):
        """ return {0, 1} sequence """
        return (np.argmax(pred_probs, axis=1) == 1).reshape(-1)

    @staticmethod
    def decode_condition(ix):
        return (ix % 2 == 1).astype(int)


class SchemeA():
    """
        Jin'scheme
            1, 0 คือ b, i ของ 1-2 พยางค์
            3, 2 คือ b, i ของ 3-4 พยางค์
            5, 4 คือ b, i ของ 5+ พยางค์ค่ะ
    """

    num_tags = 6

    @staticmethod
    def encode(labels, sy_ix, syllable_level=False):
        b_locations = np.argwhere(labels == 1).reshape(-1)

        word_boundaries = find_word_boundaries(labels)

        new_labels = []
        for wb in word_boundaries:
            num_chars = wb[1] - wb[0]
            if syllable_level:
                num_syllables = num_chars
            else:
                num_syllables = len(set(sy_ix[wb[0]:wb[1]]))

            num_inner = num_chars - 1

            if 1 <= num_syllables <= 2:
                new_labels.extend([1]+[0]*num_inner)
            elif 3 <= num_syllables <= 4:
                new_labels.extend([3]+[2]*num_inner)
            elif num_syllables >= 5:
                new_labels.extend([5]+[4]*num_inner)
            else:
                raise ValueError("Something went wrong!!")

        return np.array(new_labels).astype(int)


    @staticmethod
    def decode_condition(ix):
        return (ix % 2 == 1).astype(int)


class SchemeB():
    """
        B คือ 1, 2, 3, 4+ ค่ะ
        1 sy : [0, 1]
        2 sy : [2, 3]
        3 syl :[4, 5]
        4 + : [6, 7]
    """
    num_tags = 8

    @staticmethod
    def encode(labels, sy_ix, syllable_level=False):
        b_locations = np.argwhere(labels == 1).reshape(-1)

        word_boundaries = find_word_boundaries(labels)

        new_labels = []
        for wb in word_boundaries:
            num_chars = wb[1] - wb[0]

            if syllable_level:
                num_syllables = num_chars
            else:
                num_syllables = len(set(sy_ix[wb[0]:wb[1]]))

            ub = np.min([num_syllables, 4])*2 - 1

            if num_syllables >= 1:
                new_labels.extend([ub]+[ub-1]*(num_chars-1))
            else:
                raise ValueError("Something went wrong!!")

        return np.array(new_labels).astype(int)

    @staticmethod
    def decode_condition(ix):
        return (ix % 2 == 1).astype(int)

class SchemeASyLevel(SchemeB):
    @staticmethod
    def encode(labels, sy_ix):
        return SchemeA.encode(labels, sy_ix, syllable_level=True)

class SchemeBSyLevel(SchemeB):
    @staticmethod
    def encode(labels, sy_ix):
        return SchemeB.encode(labels, sy_ix, syllable_level=True)