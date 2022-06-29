"""Test fetch_radio_embed."""
# pylint: disable=broad-except
from fetch_radio_embed import fetch_radio_embed


def test_one_line():
    """Test one line."""
    res = fetch_radio_embed("abc")
    assert (len(res), len(res[0])) == (1, 512)


def test_two_lines():
    """Test one line."""
    res = fetch_radio_embed("abc\n test")
    assert (len(res), len(res[0])) == (2, 512)
