"""直接对比三种 viterbi 实现，重点看短串（真实场景 2-4 字占绝对多数）。

A: 原版（每步复制整条 path）
B: 通用 backpointer
C: BMES 完全展开
"""
import importlib.util
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name, root):
    pkg = os.path.join(root, "jieba")
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(pkg, "__init__.py"), submodule_search_locations=[pkg])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


orig = load("j_orig_v", os.path.join(HERE, "original"))
opt = load("j_opt_v", os.path.join(HERE, "optimized"))
F = orig.finalseg
G = opt.finalseg

start_P, trans_P, emit_P = F.start_P, F.trans_P, F.emit_P
MIN = F.MIN_FLOAT
STATES = "BMES"
PrevStatus = F.PrevStatus


def viterbi_orig(obs):
    V = [{}]
    path = {}
    for y in STATES:
        V[0][y] = start_P[y] + emit_P[y].get(obs[0], MIN)
        path[y] = [y]
    for t in range(1, len(obs)):
        V.append({})
        newpath = {}
        for y in STATES:
            em_p = emit_P[y].get(obs[t], MIN)
            prob, state = max([(V[t - 1][y0] + trans_P[y0].get(y, MIN) + em_p, y0)
                               for y0 in PrevStatus[y]])
            V[t][y] = prob
            newpath[y] = path[state] + [y]
        path = newpath
    prob, state = max((V[len(obs) - 1][y], y) for y in "ES")
    return prob, path[state]


RARE = "薬爨齉彁絵烁龘靐齾鱻麤龖厵鸜麷"
COMMON = "的一是不了人我在有他这为之大来以个中上们到说国和地也子时道出而要于就下得可你年生"


def obs_of(n, seed):
    out = []
    s = seed
    for _ in range(n):
        s = (s * 1103515245 + 12345) & 0x7FFFFFFF
        out.append(RARE[s % len(RARE)] if s % 3 else COMMON[s % len(COMMON)])
    return "".join(out)


print(f"{'n':>4} {'orig(us)':>9} {'generic':>9} {'unrolled':>9} {'gen/orig':>9} {'unr/orig':>9} {'unr/gen':>9}")
rows = []
for n in (2, 3, 4, 5, 6, 8, 12, 20, 32, 64, 128, 256):
    obs = obs_of(n, 777 + n)
    po, so = viterbi_orig(obs)
    pg, sg = G._viterbi_generic(obs, "BMES", start_P, trans_P, emit_P)
    pu, su = G._viterbi_bmes(obs)
    assert so == sg == su, (n, so, sg, su)
    assert abs(po - pg) < 1e-9 and abs(po - pu) < 1e-9

    reps = max(200, 40000 // n)
    samples = {"orig": [], "generic": [], "unrolled": []}
    for _ in range(7):
        for key, fn in (("orig", lambda: viterbi_orig(obs)),
                        ("generic", lambda: G._viterbi_generic(obs, "BMES", start_P, trans_P, emit_P)),
                        ("unrolled", lambda: G._viterbi_bmes(obs))):
            t = time.perf_counter()
            for _ in range(reps):
                fn()
            samples[key].append((time.perf_counter() - t) / reps)

    m = {k: statistics.median(v) * 1e6 for k, v in samples.items()}
    r1 = m["orig"] / m["generic"]
    r2 = m["orig"] / m["unrolled"]
    r3 = m["generic"] / m["unrolled"]
    rows.append({"n": n, "orig_us": m["orig"], "generic_us": m["generic"],
                 "unrolled_us": m["unrolled"], "generic_speedup": r1,
                 "unrolled_speedup": r2, "unrolled_vs_generic": r3})
    print(f"{n:4d} {m['orig']:9.2f} {m['generic']:9.2f} {m['unrolled']:9.2f} "
          f"{r1:8.2f}x {r2:8.2f}x {r3:8.2f}x")

with open(os.path.join(HERE, "bench_viterbi_impls.json"), "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=1)
print("\nwrote bench_viterbi_impls.json")
