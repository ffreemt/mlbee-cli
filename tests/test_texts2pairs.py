"""Test texts2pairs.

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


def test_texts2pairs_split_to_sents():
    """Test texts2pairs split to sents."""
    text1 = """Yesterday afternoon set in misty and cold. I had half a mind to spend it by my study fire, instead of wading through heath and mud to Wuthering Heights. On coming up from dinner, however (N.B. I dine between twelve and one o'clock; the housekeeper, a matronly lady, taken as a fixture along with the house, could not, or would not, comprehend my request that I might be served at five), on mounting the stairs with this lazy intention, and stepping into the room, I saw a servant girl on her knees surrounded by brushes and coal-scuttles, and raising an infernal dust as she extinguished the flames with heaps of cinders. This spectacle drove me back immediately; I took my hat, and, after a four-miles' walk, arrived at Heathcliff's garden gate just in time to escape the first feathery flakes of a snow shower.

    On that bleak hill top the earth was hard with a black frost, and the air made me shiver through every limb. Being unable to remove the chain, I jumped over, and, running up the flagged causeway bordered with straggling gooseberry bushes, knocked vainly for admittance, till my knuckles tingled and the dogs howled.
    """
    text2 = """昨天下午又冷又有雾。我想就在书房炉边消磨一下午，不想踩着杂草污泥到呼啸山庄了。

    但是，吃过午饭（注意——我在十二点与一点钟之间吃午饭，而可以当作这所房子的附属物的管家婆，一位慈祥的太太却不能，或者并不愿理解我请求在五点钟开饭的用意），在我怀着这个懒惰的想法上了楼，迈进屋子的时候，看见一个女仆跪在地上，身边是扫帚和煤斗。她正在用一堆堆煤渣封火，搞起一片弥漫的灰尘。这景象立刻把我赶回头了。我拿了帽子，走了四里路，到达了希刺克厉夫的花园口口，刚好躲过了一场今年初降的鹅毛大雪。

    在那荒凉的山顶上，土地由于结了一层黑冰而冻得坚硬，冷空气使我四肢发抖。我弄不开门链，就跳进去，顺着两边种着蔓延的醋栗树丛的石路跑去。我白白地敲了半天门，一直敲到我的手指骨都痛了，狗也狂吠起来。

    """
    url = "https://hf.space/embed/mikeee/radio-mlbee/+/api/predict/"
    url_dev = "https://hf.space/embed/mikeee/radio-mlbee-dev/+/api/predict/"
    res = texts2pairs(text1, text2, url=url_dev, split_to_sents=True)
    assert len(res) == 9

    # 7
    # Being unable to remove the chain, I jumped over, and, running up the flagged causeway bordered with straggling gooseberry bushes, knocked vainly for admittance, till my knuckles tingled and the dogs howled.
    # 我弄不开门链，就跳进去，顺着两边种着蔓延的醋栗树丛的石路跑去。
    # 0.64

    assert "jump" in res[7][0]
    assert "跳" in res[7][1]
    assert res[-2][2] > 0.6
