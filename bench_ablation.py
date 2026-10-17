"""四臂归因基准：A 原版 / B 仅 _route / C 仅 viterbi / D 两者。

用途：确认两处改动各自贡献了多少，避免"两个改动打包进一个基准"导致归因错误。

⚠ 这个基准必须**跨进程**跑（同名包无法在同一进程共存），因此噪声明显：
  本机实测连不该变的 `cut_all` 对照组都跑出过 0.84x。所以这张表只用来读
  **趋势与归属**，绝对数值请以 bench_final.py（同进程、低噪声）为准。

B / C 两个变体目录会由本脚本自动重建：从 optimized/ 出发，用 git checkout
恢复单个文件即可。
"""
import json
import os
import shutil
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
OPT = os.path.join(HERE, "optimized")
ORIG = os.path.join(HERE, "original")
PATCH = os.path.join(HERE, "jieba-optimization.patch")

RUNNER = r'''
import json, sys, time
root, corpus_path, mode, reps = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
sys.path.insert(0, root)
import jieba
import jieba.posseg
jieba.initialize()
with open(corpus_path, encoding="utf-8") as f:
    data = json.load(f)
docs = data["docs"]
text = "".join(docs)
rare = "薬爨齉彁絵烁龘靐齾鱻麤龖厵鸜麷"
heavy = "".join(rare[(i * 7 + j) % len(rare)] for i in range(120) for j in range(40))

def bench(fn):
    best = 1e9
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        dt = time.perf_counter() - t
        if dt < best:
            best = dt
    return best

if mode == "cut":
    def f():
        for d in docs:
            for _ in jieba.cut(d): pass
elif mode == "nohmm":
    def f():
        for d in docs:
            for _ in jieba.cut(d, HMM=False): pass
elif mode == "search":
    def f():
        for d in docs:
            for _ in jieba.cut_for_search(d): pass
elif mode == "cutall":
    def f():
        for d in docs:
            for _ in jieba.cut(d, cut_all=True): pass
elif mode == "whole":
    def f():
        for _ in jieba.cut(text): pass
elif mode == "hmm_heavy":
    def f():
        for _ in range(12):
            for _ in jieba.cut(heavy): pass
else:
    raise SystemExit("bad mode")

f()
print(json.dumps({"t": bench(f)}))
'''


def run(root, corpus, mode, reps=3):
    p = subprocess.run([PY, "-c", RUNNER, root, corpus, mode, str(reps)],
                       capture_output=True, text=True, encoding="utf-8")
    if p.returncode != 0:
        raise RuntimeError(f"{root} {mode}:\n{p.stderr[-1500:]}")
    line = [l for l in p.stdout.strip().splitlines() if l.startswith("{")][-1]
    return json.loads(line)["t"]


def build_variants():
    """B = 只改 jieba/__init__.py；C = 只改 finalseg/__init__.py。"""
    b = os.path.join(HERE, "variant_route_only")
    c = os.path.join(HERE, "variant_viterbi_only")
    for d in (b, c):
        shutil.rmtree(d, ignore_errors=True)
        shutil.copytree(OPT, d)
        shutil.rmtree(os.path.join(d, ".git"), ignore_errors=True)

    # 反向补丁只作用于单个文件：先重建原版对应文件，再覆盖回变体
    tmp = os.path.join(HERE, "_tmp_revert")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(OPT, tmp)
    shutil.rmtree(os.path.join(tmp, ".git"), ignore_errors=True)
    r = subprocess.run(["git", "-c", "core.autocrlf=false", "apply", "-R", "-p1",
                        "--whitespace=nowarn", PATCH], cwd=tmp,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("reverse patch failed: " + r.stderr)

    # B 恢复 finalseg，C 恢复 jieba/__init__.py
    shutil.copy2(os.path.join(tmp, "jieba", "finalseg", "__init__.py"),
                 os.path.join(b, "jieba", "finalseg", "__init__.py"))
    shutil.copy2(os.path.join(tmp, "jieba", "__init__.py"),
                 os.path.join(c, "jieba", "__init__.py"))
    shutil.rmtree(tmp, ignore_errors=True)
    return b, c


ARMS = {"A_orig": ORIG}


def main():
    corpus = os.path.join(HERE, "corpus.json")
    b, c = build_variants()
    ARMS["B_route"] = b
    ARMS["C_viterbi"] = c
    ARMS["D_both"] = OPT

    modes = ["cut", "nohmm", "search", "cutall", "whole", "hmm_heavy"]
    rounds = int(os.environ.get("ROUNDS", "7"))
    out = {}

    for mode in modes:
        samples = {k: [] for k in ARMS}
        for _ in range(rounds):
            for name, root in ARMS.items():
                samples[name].append(run(root, corpus, mode))
        base = samples["A_orig"]
        row = {"A_ms": statistics.median(base) * 1000}
        for name in ("B_route", "C_viterbi", "D_both"):
            ratios = [x / y for x, y in zip(base, samples[name])]
            row[name] = {
                "ms": statistics.median(samples[name]) * 1000,
                "speedup": statistics.median(ratios),
                "lo": min(ratios), "hi": max(ratios),
            }
        out[mode] = row
        bb, cc, dd = row["B_route"], row["C_viterbi"], row["D_both"]
        print(f"{mode:10s} A {row['A_ms']:7.2f}ms | "
              f"B {bb['speedup']:.2f}x | C {cc['speedup']:.2f}x | D {dd['speedup']:.2f}x")

    with open(os.path.join(HERE, "bench_ablation.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\nwrote bench_ablation.json (cross-process, noisy: read trends only)")


if __name__ == "__main__":
    sys.exit(main())
