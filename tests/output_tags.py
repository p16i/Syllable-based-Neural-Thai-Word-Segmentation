import pytest
import numpy as np

from attacut import output_tags

# Note: elements in sy_ix are dummy values.
@pytest.mark.parametrize(
    ("labels", "sy_ix", "expected"),
    [
        ( # character sequence
            [1, 0, 0, 0, 0], [7, 7, 7, 8, 8],
            {
                "BI": [1, 0, 0, 0, 0],
                "SchemeA": [1, 0, 0, 0, 0],
                "SchemeB": [3, 2, 2, 2, 2]
            }
        ),
        ( # character sequence
            [1, 0, 0, 0, 0, 0, 0], [7, 7, 7, 8, 8, 9, 9],
            {
                "BI": [1, 0, 0, 0, 0, 0, 0],
                "SchemeA": [3, 2, 2, 2, 2, 2, 2],
                "SchemeB": [5, 4, 4, 4, 4, 4, 4]
            }
        ),
        ( # character sequence
            [1, 0, 0, 0, 0, 1, 0], [7, 7, 7, 8, 8, 9, 9],
            {
                "BI": [1, 0, 0, 0, 0, 1, 0],
                "SchemeA": [1, 0, 0, 0, 0, 1, 0],
                "SchemeB": [3, 2, 2, 2, 2, 1, 0]
            }
        ),
        ( # syllable sequence
            [1, 0, 1, 1, 1, 0, 0], [7, 8, 9, 2, 1, 3],
            {
                "BI": [1, 0, 1, 1, 1, 0, 0],
                "SchemeASyLevel": [1, 0, 1, 1, 3, 2, 2],
                "SchemeBSyLevel": [3, 2, 1, 1, 5, 4, 4],
            }
        ),
        ( # syllable sequence
            [1, 1, 0, 0, 0], [7, 3, 1, 2, 2],
            {
                "BI": [1, 1, 0, 0, 0],
                "SchemeASyLevel": [1, 3, 2, 2, 2],
                "SchemeBSyLevel": [1, 7, 6, 6, 6],
            }
        )
    ]
)
def test_encode(labels, sy_ix, expected):

    for name, exp in expected.items():
        scheme = output_tags.get_scheme(name)

        np.testing.assert_array_equal(scheme.encode(labels, sy_ix), exp)

@pytest.mark.parametrize(
    ("preds", "expected"),
    [
        ( # character sequence
            {
                "BI": [1, 0, 0, 0, 0],
                "SchemeA": [1, 0, 0, 0, 0],
                "SchemeB": [1, 0, 0, 0, 0]
            },
            [1, 0, 0, 0, 0]
        ),
        ( # character sequence
            {
                "BI": [1, 0, 1, 0, 0, 0, 0, 0], # pretend  the last word has three syllables
                "SchemeA": [1, 0, 3, 2, 2, 2, 2, 2],
                "SchemeB": [1, 0, 5, 4, 4, 4, 4, 4]
            },
            [1, 0, 1, 0, 0, 0, 0, 0]
        ),
        ( # syllable sequence
            {
                "BI": [1, 0, 1, 0, 0],
                "SchemeASyLevel": [1, 0, 3, 2, 2],
                "SchemeBSyLevel": [3, 2, 5, 4, 4]
            },
            [1, 0, 1, 0, 0]
        ),
        ( # syllable sequence
            {
                "BI": [1, 0, 1, 0, 0, 1, 0, 0, 0],
                "SchemeASyLevel": [1, 0, 3, 2, 2, 5, 4, 4, 4],
                "SchemeBSyLevel": [3, 2, 5, 4, 4, 7, 6, 6, 6]
            },
            [1, 0, 1, 0, 0, 1, 0, 0, 0]
        ),
    ]
)
def test_decode(preds, expected):
    for name, pred in preds.items():
        scheme = output_tags.get_scheme(name)

        np.testing.assert_array_equal(
            scheme.decode_condition(np.array(pred)),
            expected
        )

@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        (
            (1, 0, 0, 1, 0), [(0, 3), (3, 5)]
        ),
        (
            (1, 0, 0, 0, 0), [(0, 5)]
        ),
        (
            (1, 0, 0, 0, 0), [(0, 5)]
        ),
        (
            (1, 0, 1, 0, 0, 1, 0, 0), [(0, 2), (2, 5), (5, 8)]
        ),
    ]
)
def test_find_word_boundaries(labels, expected):
    boundaries = output_tags.find_word_boundaries(labels)

    assert boundaries == expected