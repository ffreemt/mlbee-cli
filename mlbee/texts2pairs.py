"""Align two texts."""
# pylint=disable=invalid-name
from typing import List

import httpx
from logzero import logger

url = "https://hf.space/embed/mikeee/radio-mlbee/+/api/predict/"

# 30s timeout on connect, no timeout elsewhere
timeout = httpx.Timeout(None, connect=30)


def texts2pairs(
    text1: str,
    text2: str,
    url: str = url,
    timeout: httpx.Timeout = timeout,
) -> List:
    r"""Sent texts to url.

    Args:
        text1: text (str)
        text2: text (str)
        url: service
        timeout: default connect=10s, None elesewhere

    text1 = "test1\n a b c\nlove"; text2 = "测试\n爱"
    """
    try:
        resp = httpx.post(
            url,
            json={"data": [text1, text2]},
            timeout=timeout,
        )
        resp.raise_for_status()
        texts2pairs.resp = resp  # save the whole thing
    except Exception as exc:
        logger.error(exc)
        raise
    try:
        jdata = resp.json()
    except Exception as exc:
        logger.error(exc)
        raise

    # # {'data': [{'headers': ['text1', 'text2', 'llh'],
    # 'data': [['tes1', '测试', 0.36],
    # ['a b c\\love', '爱', 0.67]]}],
    # 'duration': 0.25752758979797363,
    # 'average_duration': 0.257527589797973 63}
    try:
        _ = jdata.get("data")[0].get("data")
    except Exception as exc:
        logger.error(exc)
        raise
    return _
