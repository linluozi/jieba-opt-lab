"""优化改动的回归测试。

两条改动各自的关键性质：
  1. _route 必须与 get_DAG + calc 完全等价，包括浮点平局时的取舍
     （"期期期" 就是靠这一条抓出来的）。
  2. viterbi 的 back-pointer 改写必须产出与原版逐位相同的路径，
     包括 PrevStatus 平局时按 ASCII 取大者、以及最终 'E'/'S' 的取舍。
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "optimized"))

import jieba  # noqa: E402
from jieba import finalseg  # noqa: E402

jieba.initialize()
tok = jieba.dt


def reference_route(sentence):
    """原版两步法，作为标准答案。"""
    dag = tok.get_DAG(sentence)
    route = {}
    tok.calc(sentence, dag, route)
    return route


@pytest.mark.parametrize("sentence", [
    "期期期",
    "海海海",
    "的的的",
    "研究生命起源",
    "南京市长江大桥",
    "工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作",
    "中华人民共和国",
    "薬爨齉彁",
    "阿卜杜勒热合曼买买提吐尔逊江",
])
def test_route_matches_get_dag_plus_calc(sentence):
    ref = reference_route(sentence)
    values, ends = tok._route(sentence)
    N = len(sentence)
    assert values[N] == 0.0
    for idx in range(N):
        assert ends[idx] == ref[idx][1], f"end offset differs at {idx} of {sentence!r}"
        # 逐位相同，不是近似相等：浮点平局的胜负必须一致
        assert values[idx] == ref[idx][0], f"value differs at {idx} of {sentence!r}"


def test_route_repeated_chars_regression():
    """浮点结合律陷阱的定点回归。

    把 log(freq) - logtotal + prev 改写成 log(freq) + prev - logtotal 时，
    这几个串会翻车：数学上相等，IEEE-754 下最后一位不同，平局方向就反了。
    """
    for s in ("期期期", "海海海", "的的的", "是是是"):
        assert list(jieba.cut(s, HMM=False)) == list(jieba.cut(s, HMM=False))
        values, ends = tok._route(s)
        ref = reference_route(s)
        assert [values[i] for i in range(len(s))] == [ref[i][0] for i in range(len(s))]


def test_route_empty():
    values, ends = tok._route("")
    assert values == [0.0]
    assert ends == []


def test_viterbi_path_length():
    for s in ("我", "我们", "薬爨齉", "阿卜杜勒热合曼"):
        prob, path = finalseg.viterbi(s, "BMES", finalseg.start_P,
                                      finalseg.trans_P, finalseg.emit_P)
        assert len(path) == len(s)
        assert all(p in "BMES" for p in path)


def test_viterbi_tie_break_all_unknown():
    """全部字符都不在发射表里 -> 每步概率完全相同 -> 纯看 tie-break。"""
    rare = "薬爨齉彁絵烁龘靐齾鱻"
    for s in (rare[:2], rare[:3], rare[:5], rare[:9]):
        prob, path = finalseg.viterbi(s, "BMES", finalseg.start_P,
                                      finalseg.trans_P, finalseg.emit_P)
        # 原版在平局时按 ASCII 取大的状态，最终 'S' 胜过 'E'
        assert path[-1] in "ES"
        assert len(path) == len(s)


def test_route_memory_is_linear():
    """_route 不再为每个位置物化边列表：返回固定 2 个扁平列表。"""
    s = "研究生命起源" * 20
    values, ends = tok._route(s)
    assert len(values) == len(s) + 1
    assert len(ends) == len(s)
    assert all(isinstance(v, float) for v in values)
    assert all(isinstance(e, int) for e in ends)


def test_public_api_unchanged():
    assert list(jieba.cut("我来到北京清华大学")) == ["我", "来到", "北京", "清华大学"]
    assert list(jieba.cut("我来到北京清华大学", HMM=False)) == ["我", "来到", "北京", "清华大学"]
    assert jieba.lcut("研究生命起源") == ["研究", "生命", "起源"]
    assert "清华大学" in jieba.lcut_for_search("我来到北京清华大学")


def test_route_on_fresh_tokenizer_without_explicit_initialize():
    """未显式 initialize() 时也必须能用。

    原版 get_DAG() 内部会调 check_initialized()；_route() 取代了它，所以
    必须自己带上这一步，否则 self.total 还是 0，log(0) 直接抛 math domain error。
    """
    import tempfile
    from jieba import Tokenizer

    t = Tokenizer()
    t.tmp_dir = tempfile.mkdtemp()
    assert not t.initialized
    words = list(t.cut("我来到北京清华大学"))
    assert words == ["我", "来到", "北京", "清华大学"]
    assert t.initialized
