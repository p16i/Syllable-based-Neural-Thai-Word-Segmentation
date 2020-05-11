import numpy as np
import pytest

from attacut import char_type

@pytest.mark.parametrize(
    ("seq", "expected"),
    [
        ("ผมไม่", ("n", "c", "w", "c", "t")),
        ("รู้มาก", ("c", "v", "t", "c", "v", "c")),
        ("ที่แล้วมา", ("c", "v", "t", "w", "c", "t", "c", "c", "v")),
        ("ปรากฏการ CoViD", ("c", "c", "v", "c", "c", "c", "v", "c", "p", "b_e", "s_e", "b_e", "s_e", "b_e")),
    ]
)
def test_something(seq, expected):
    ch_type = char_type.get_char_type_cat(
        char_type.get_char_type_ix(seq)
    )
    assert len(ch_type) == len(expected)
    assert tuple(ch_type) == expected
