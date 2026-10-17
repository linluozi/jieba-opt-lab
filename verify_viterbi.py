"""viterbi 专项等价测试：直接在函数层面穷举，覆盖平局分支。

`期期期` 那类不一致属于 route 的浮点舍入问题，已在 _route 里修掉；
这个脚本单独把 viterbi 拉出来，用穷举 + 随机覆盖所有 tie 情况，
确保 back-pointer 改写没有改变任何一条路径。
"""
import importlib
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

CASES = r'''
import json, sys, itertools, random
sys.path.insert(0, sys.argv[1])
from jieba import finalseg
out = []
# 观测集合：常用字 + 完全不在 emit 表的字符（保证全 MIN_FLOAT，最大化平局）
common = "的一是不了人我在有他这为之大来以个中上们到说国和地也子时道出而要于就下得可你年生"
rare = "薬爨齉彁絵烁龘靐齾鱻"
alpha = common + rare

# 1. 全 1/2/3 长度穷举（单字和双字的所有组合）
for ch in alpha:
    out.append(ch)
for a in alpha[:40]:
    for b in alpha[:40]:
        out.append(a + b)

# 2. 三字穷举（rare 之间，最容易平局）
for a in rare:
    for b in rare:
        for c in rare:
            out.append(a + b + c)

# 3. 随机长度 4..30
rng = random.Random(4242)
for _ in range(4000):
    L = rng.randint(4, 30)
    out.append("".join(rng.choice(alpha) for _ in range(L)))

# 4. 长串（O(n^2) 复制压力）
for L in (50, 80, 120, 200, 300):
    out.append("".join(rng.choice(rare) for _ in range(L)))

res = []
for s in out:
    prob, path = finalseg.viterbi(s, 'BMES', finalseg.start_P, finalseg.trans_P, finalseg.emit_P)
    res.append([prob, "".join(path)])
print(json.dumps(res, ensure_ascii=False))
'''


def run(root):
    p = subprocess.run([PY, "-c", CASES, root], capture_output=True, text=True, encoding="utf-8")
    if p.returncode != 0:
        print("FAILED", root, p.stderr[-2000:])
        sys.exit(1)
    return json.loads(p.stdout.strip().splitlines()[-1])


import json  # noqa: E402


def main():
    a = run(os.path.join(HERE, "original"))
    b = run(os.path.join(HERE, "optimized"))
    print("viterbi cases:", len(a))
    bad = 0
    for i, (x, y) in enumerate(zip(a, b)):
        if x[1] != y[1] or abs(x[0] - y[0]) > 1e-9:
            bad += 1
            if bad <= 5:
                print(f"  DIFF #{i}: prob {x[0]} vs {y[0]}")
                print(f"    orig {x[1][:80]}")
                print(f"    opt  {y[1][:80]}")
    print(f"mismatches: {bad}")
    print("RESULT:", "ALL IDENTICAL" if bad == 0 else "MISMATCH")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
