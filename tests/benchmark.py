import numpy as np
import pytest

from attacut import benchmark


@pytest.mark.parametrize(
    ("txt", "expected"),
    [
        ("วันนี้|ทำไม|", "วันนี้|ทำไม"),
        ("a|dog|", "a|dog"),
        ("<NE>Peter</NE>", "Peter"),
        ("abc| | ", "abc"),
        ("foo| ||| ", "foo"),
    ]
)
def test_preprocessing(txt, expected):
    assert benchmark.preprocessing(txt) == expected