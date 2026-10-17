"""等价性验证：原版 vs 优化版，所有分词模式逐字节比对。
两版各用独立子进程导入（同进程会互相覆盖 sys.modules）。
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

DUMP = r'''
import json, os, sys
sys.path.insert(0, sys.argv[1])
import jieba
from jieba import finalseg
jieba.initialize()
with open(sys.argv[2], encoding="utf-8") as f:
    payload = json.load(f)
docs = payload["docs"]
extra = payload["extra"]
out = {}
def run(fn, items):
    return [list(fn(x)) for x in items]
out["cut"] = run(jieba.cut, docs)
out["cut_nohmm"] = run(lambda s: jieba.cut(s, HMM=False), docs)
out["cut_all"] = run(lambda s: jieba.cut(s, cut_all=True), docs)
out["cut_for_search"] = run(jieba.cut_for_search, docs)
out["lcut"] = [jieba.lcut(x) for x in extra]
out["tokenize"] = [list(jieba.tokenize(x)) for x in extra]
out["tokenize_search"] = [list(jieba.tokenize(x, mode="search")) for x in extra]
out["posseg"] = [[(w, t) for w, t in __import__("jieba.posseg", fromlist=["x"]).cut(x)] for x in extra]
out["finalseg"] = [list(finalseg.cut(x)) for x in extra]
out["dag"] = [{str(k): v for k, v in jieba.dt.get_DAG(x).items()} for x in extra]
print(json.dumps(out, ensure_ascii=False))
'''


def dump(root, payload_path):
    p = subprocess.run([PY, "-c", DUMP, root, payload_path],
                       capture_output=True, text=True, encoding="utf-8")
    if p.returncode != 0:
        print("FAILED", root)
        print(p.stderr[-3000:])
        sys.exit(1)
    return json.loads(p.stdout.strip().splitlines()[-1])


def main():
    with open(os.path.join(HERE, "corpus.json"), encoding="utf-8") as f:
        corpus = json.load(f)

    # 补充边界用例：空串、单字、纯符号、英文混排、长未登录串、生僻字
    extra = [
        "",
        "我",
        "。",
        "，，，，",
        "abc",
        "Python和C++",
        "24口交换机",
        "阿卜杜勒热合曼买买提吐尔逊江",
        "阿卜杜勒热合曼买买提吐尔逊江" * 3,
        "薬爨齉彁絵烁",
        "这是一段测试文本，包含中文、English、123数字和符号！",
        "中华人民共和国中央人民政府",
        "工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作",
        "研究生命起源",
        "长春市长春节讲话",
        "他说的确实在理",
        "结婚的和尚未结婚的",
        "武汉市长江大桥",
        "南京市长 江大桥",
        "北京大学生前来应聘",
    ]
    payload = {"docs": corpus["docs"], "extra": extra}
    pp = os.path.join(HERE, "_equiv_payload.json")
    with open(pp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    a = dump(os.path.join(HERE, "original"), pp)
    b = dump(os.path.join(HERE, "optimized"), pp)

    keys = sorted(a)
    ok = True
    for k in keys:
        if a[k] == b[k]:
            n = len(a[k])
            print(f"  {k:18s} identical  ({n} cases)")
        else:
            ok = False
            print(f"  {k:18s} DIFFERS")
            for i, (x, y) in enumerate(zip(a[k], b[k])):
                if x != y:
                    print(f"    first diff at #{i}")
                    print(f"      orig: {str(x)[:300]}")
                    print(f"      opt : {str(y)[:300]}")
                    break
    print("\nRESULT:", "ALL IDENTICAL" if ok else "MISMATCH FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
