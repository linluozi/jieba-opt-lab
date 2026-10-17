"""最终基准：每个 mode 的负载都调到 >=5ms，避免计时器量化。
轮内交错 + best-of-3 + 逐轮比值中位数。
"""
import importlib.util
import json
import logging
import os
import statistics
import sys
import time


def load_pkg(name, root):
    pkg_dir = os.path.join(root, "jieba")
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(pkg_dir, "__init__.py"),
        submodule_search_locations=[pkg_dir])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


HERE = os.path.dirname(os.path.abspath(__file__))
A = load_pkg("jieba_orig_f", os.path.join(HERE, "original"))
B = load_pkg("jieba_opt_f", os.path.join(HERE, "optimized"))
for m in (A, B):
    m.setLogLevel(logging.WARNING)
    m.initialize()

with open(os.path.join(HERE, "corpus.json"), encoding="utf-8") as f:
    DOCS = json.load(f)["docs"]
TEXT = "".join(DOCS)
RARE = "薬爨齉彁絵烁龘靐齾鱻麤龖厵鸜麷"
# 长未登录串：HMM 主导，且长度足够体现 O(n^2) 差异
HEAVY_DOC = "".join(RARE[(i * 7 + j) % len(RARE)] for i in range(120) for j in range(40))
HEAVY = [HEAVY_DOC] * 12

for d in DOCS[:5]:
    assert list(A.cut(d)) == list(B.cut(d))
assert list(A.cut(HEAVY_DOC)) == list(B.cut(HEAVY_DOC))
print("sanity ok, HEAVY_DOC len =", len(HEAVY_DOC), "\n")


def tasks(mod):
    return {
        "cut": lambda: [w for d in DOCS for w in mod.cut(d)],
        "nohmm": lambda: [w for d in DOCS for w in mod.cut(d, HMM=False)],
        "search": lambda: [w for d in DOCS for w in mod.cut_for_search(d)],
        "cutall": lambda: [w for d in DOCS for w in mod.cut(d, cut_all=True)],
        "whole": lambda: list(mod.cut(TEXT)),
        "hmm_heavy": lambda: [w for d in HEAVY for w in mod.cut(d)],
    }


TA, TB = tasks(A), tasks(B)


def best_of(fn, reps=3):
    best = 1e9
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        dt = time.perf_counter() - t
        if dt < best:
            best = dt
    return best


ROUNDS = 15
res = {}
for mode in TA:
    TA[mode](); TB[mode]()
    ta, tb = [], []
    for _ in range(ROUNDS):
        ta.append(best_of(TA[mode]))
        tb.append(best_of(TB[mode]))
    ratios = [x / y for x, y in zip(ta, tb)]
    res[mode] = {
        "orig_ms": statistics.median(ta) * 1000,
        "opt_ms": statistics.median(tb) * 1000,
        "speedup": statistics.median(ratios),
        "lo": min(ratios),
        "hi": max(ratios),
    }
    r = res[mode]
    print(f"{mode:10s} {r['orig_ms']:8.2f} -> {r['opt_ms']:8.2f} ms | "
          f"{r['speedup']:6.3f}x [{r['lo']:.3f}, {r['hi']:.3f}]")

with open(os.path.join(HERE, "bench_final.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print("\nwrote bench_final.json")
