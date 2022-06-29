"""Align two texts."""
# pylint=disable=invalid-name
from typing import List, Optional

import httpx
from logzero import logger
from tenacity import retry
from tenacity.stop import stop_after_attempt, stop_after_delay

url_hf = "https://hf.space/embed/mikeee/radio-mlbee/+/api/predict/"

# 60s timeout on connect, no timeout elsewhere
timeout_ = httpx.Timeout(None, connect=60)


@retry(stop=(stop_after_delay(10) | stop_after_attempt(5)))
def texts2pairs(
    text1: str,
    text2: str,
    split_to_sents: bool = False,
    url: Optional[str] = None,
    timeout: httpx.Timeout = timeout_,
) -> List:
    r"""Sent texts to url for aligning.

    Args:
        text1: text (str)
        text2: text (str)
        url: service api, if None, set to default url_hf
        timeout: default connect=10s, None elesewhere
        split_to_sents: split text to sents when True, default False

    text1 = "test1\n a b c\nlove"; text2 = "测试\n爱"
    """
    if url is None:
        url = url_hf
    try:
        resp = httpx.post(
            url,
            # json={"data": [text1, text2]},
            json={
                "data": [text1, text2, split_to_sents, False]
            },  # 4th param: preview, always set to False
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
