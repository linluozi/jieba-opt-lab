"""模糊测试：随机生成大量字符串，原版 vs 优化版逐字节比对。
重点覆盖：
  - 生僻字/未登录字（触发 viterbi 平局分支）
  - 极短串（len 1-3，viterbi 边界）
  - 长未登录串（O(n^2) 复制路径）
  - 中英数符号混排
"""
import json
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

FUZZ = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
import jieba
import jieba.posseg
jieba.initialize()
with open(sys.argv[2], encoding="utf-8") as f:
    items = json.load(f)
out = []
for s in items:
    out.append([
        list(jieba.cut(s)),
        list(jieba.cut(s, HMM=False)),
        list(jieba.cut(s, cut_all=True)),
        list(jieba.cut_for_search(s)),
        list(jieba.tokenize(s)),
        [(w, t) for w, t in jieba.posseg.cut(s)],
    ])
print(json.dumps(out, ensure_ascii=False))
'''


def dump(root, items_path):
    p = subprocess.run([PY, "-c", FUZZ, root, items_path],
                       capture_output=True, text=True, encoding="utf-8")
    if p.returncode != 0:
        print("FAILED", root, p.stderr[-2000:])
        sys.exit(1)
    return json.loads(p.stdout.strip().splitlines()[-1])


def build_items():
    rng = random.Random(20261018)
    items = []

    # 1. 词典高频字组成的随机串
    with open(os.path.join(HERE, "corpus.json"), encoding="utf-8") as f:
        docs = json.load(f)["docs"]
    pool = "".join(docs)
    for _ in range(1500):
        L = rng.randint(1, 60)
        items.append("".join(rng.choice(pool) for _ in range(L)))

    # 2. 生僻字 / 未登录字（viterbi 里全部落 MIN_FLOAT，最容易暴露平局差异）
    rare = "薬爨齉彁絵烁龘靐齾鱻麤龖厵鸜麷饢籱鬮"
    for _ in range(1200):
        L = rng.randint(1, 40)
        items.append("".join(rng.choice(rare) for _ in range(L)))

    # 3. 生僻字 + 常用字混合
    for _ in range(800):
        L = rng.randint(2, 30)
        items.append("".join(rng.choice(rare if rng.random() < 0.5 else pool)
                             for _ in range(L)))

    # 4. 中英数符号混排
    alphabet = "abcXYZ019+#&._%- ，。！\t\n"
    for _ in range(600):
        L = rng.randint(0, 40)
        items.append("".join(rng.choice(alphabet) for _ in range(L)))

    # 5. 单字与空串
    items.extend(["", " ", "\n", "a", "我", "。", "薬"])
    for ch in pool[:60]:
        items.append(ch)
        items.append(ch * 2)
        items.append(ch * 3)

    # 6. 长未登录串（O(n^2) 复制压力点）
    for L in (50, 100, 200, 400):
        items.append("".join(rng.choice(rare) for _ in range(L)))

    return items


def main():
    items = build_items()
    print("fuzz cases:", len(items))
    pp = os.path.join(HERE, "_fuzz_payload.json")
    with open(pp, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False)

    a = dump(os.path.join(HERE, "original"), pp)
    b = dump(os.path.join(HERE, "optimized"), pp)

    bad = 0
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            bad += 1
            if bad <= 5:
                print(f"\nDIFF at case #{i}: {items[i]!r}")
                for j, name in enumerate(["cut", "nohmm", "cut_all", "search", "tokenize", "posseg"]):
                    if x[j] != y[j]:
                        print(f"  mode={name}")
                        print(f"    orig: {str(x[j])[:200]}")
                        print(f"    opt : {str(y[j])[:200]}")
    print(f"\ncases={len(a)}  mismatches={bad}")
    print("RESULT:", "ALL IDENTICAL" if bad == 0 else "MISMATCH")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
