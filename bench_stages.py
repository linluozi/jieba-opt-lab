"""测量精确模式 cut() 内部的阶段耗时占比，用于缺陷分析图。

只测原版：把 get_DAG / calc / finalseg(HMM) 拆开单独计时，
再用「掏空法」验证：把 calc 换成空壳，看整体掉多少。
"""
import json
import logging
import os
import sys
import time
from math import log

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "original"))

import jieba  # noqa: E402
from jieba import finalseg  # noqa: E402

jieba.setLogLevel(logging.WARNING)
jieba.initialize()
tok = jieba.dt

with open(os.path.join(HERE, "corpus.json"), encoding="utf-8") as f:
    docs = json.load(f)["docs"]
text = "".join(docs)
blocks = [b for b in jieba.re_han_default.split(text)
          if b and jieba.re_han_default.match(b)]
print("blocks:", len(blocks), "chars:", sum(len(b) for b in blocks))


def best_of(fn, n=9):
    best = 1e9
    for _ in range(n):
        t = time.perf_counter()
        fn()
        dt = time.perf_counter() - t
        if dt < best:
            best = dt
    return best


def only_dag():
    for b in blocks:
        tok.get_DAG(b)


def dag_plus_calc():
    for b in blocks:
        dag = tok.get_DAG(b)
        tok.calc(b, dag, {})


def full():
    for d in docs:
        for _ in tok.cut(d):
            pass


def full_nohmm():
    for d in docs:
        for _ in tok.cut(d, HMM=False):
            pass


t_dag = best_of(only_dag)
t_dagcalc = best_of(dag_plus_calc)
t_full = best_of(full)
t_nohmm = best_of(full_nohmm)

# 掏空法：把 calc 换成空壳（保留 dict 写入形状，防 DCE）
orig_calc = tok.calc


def hollow_calc(sentence, DAG, route):
    route[len(sentence)] = (0, 0)
    for idx in DAG:
        route[idx] = (0.0, DAG[idx][0])


tok.calc = hollow_calc
t_hollow = best_of(full)
tok.calc = orig_calc

stages = {
    "get_DAG": t_dag,
    "calc": t_dagcalc - t_dag,
    "HMM (finalseg)": t_full - t_nohmm,
    "cut 其余（正则/组装）": t_full - t_dagcalc - (t_full - t_nohmm),
}
print(f"\nfull cut()      : {t_full*1000:7.2f} ms")
print(f"full HMM=False  : {t_nohmm*1000:7.2f} ms")
print(f"get_DAG only    : {t_dag*1000:7.2f} ms")
print(f"get_DAG + calc  : {t_dagcalc*1000:7.2f} ms")
print(f"calc hollowed   : {t_hollow*1000:7.2f} ms  (sanity: 应接近 HMM=False)")
print()
total = sum(v for v in stages.values())
for k, v in stages.items():
    print(f"  {k:22s} {v*1000:7.2f} ms  {v/t_full*100:5.1f}% of full")

out = {
    "full_ms": t_full * 1000,
    "full_nohmm_ms": t_nohmm * 1000,
    "get_dag_ms": t_dag * 1000,
    "dag_calc_ms": t_dagcalc * 1000,
    "hollow_calc_ms": t_hollow * 1000,
    "stages_ms": {k: v * 1000 for k, v in stages.items()},
    "stages_pct": {k: v / t_full * 100 for k, v in stages.items()},
}
with open(os.path.join(HERE, "bench_stages.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("\nwrote bench_stages.json")
