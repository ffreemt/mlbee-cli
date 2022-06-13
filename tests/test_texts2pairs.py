"""Test texts2pairs

{
 'data': [
    {'data': [['test1', '测试', 0.76],
                    ['a b c', '', ''],
                    ['love', '爱', 0.96]],
      'headers': ['text1', 'text2', 'llh']
    }
  ],
 'duration': 0.048053741455078125
 'average_duration': 0.33365621286280017,
}
"""
from logzero import logger

from mlbee.texts2pairs import texts2pairs


def test_texts2pairs_minimal():
    """Test texts2pairs minimal."""
    text1 = "test1\n a b c\nlove"
    text2 = "测试\n爱"
    try:
        res = texts2pairs(text1, text2)
    except Exception as exc:
        logger.error(exc)
        raise
    assert res[2][1] == "爱"
    assert res[2][2] > 0.9
