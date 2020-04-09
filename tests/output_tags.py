import pytest

from attacut import output_tags

# Note: elements in sy_ix are dummy values.
@pytest.mark.parametrize(
    ("labels", "sy_ix", "expected"),
    [
        (
            [1, 0, 0, 0, 0], [7, 7, 7, 8, 8],
            {
                "BI": [1, 0, 0, 0, 0],
                "SchemeA": [1, 0, 0, 0, 0],
                "SchemeB": [3, 2, 2, 2, 2]
            }
        ),
        (
            [1, 0, 0, 0, 0, 0, 0], [7, 7, 7, 8, 8, 9, 9],
            {
                "BI": [1, 0, 0, 0, 0, 0, 0],
                "SchemeA": [3, 2, 2, 2, 2, 2, 2],
                "SchemeB": [5, 4, 4, 4, 4, 4, 4]
            }
        ),
        (
            [1, 0, 0, 0, 0, 1, 0], [7, 7, 7, 8, 8, 9, 9],
            {
                "BI": [1, 0, 0, 0, 0, 1, 0],
                "SchemeA": [1, 0, 0, 0, 0, 1, 0],
                "SchemeB": [3, 2, 2, 2, 2, 1, 0]
            }
        )
    ]
)
def test_encode(labels, sy_ix, expected):

    for name, exp in expected.items():
        scheme = output_tags.get_scheme(name)

        assert scheme.encode(labels, sy_ix) == exp

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