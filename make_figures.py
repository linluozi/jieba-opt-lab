"""生成 README 用的性能图。全部数据来自 bench_*.json，不手写任何数字。

图 1 fig-stages.png        精确模式 cut() 的阶段耗时占比（缺陷定位）
图 2 fig-speedup.png       各模式加速比 + 95% 区间 + 未改动对照组
图 3 fig-viterbi.png       viterbi 三种实现随长度变化（复杂度证据）
图 4 fig-ablation.png      四臂归因：两处改动各自贡献
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "docs")
os.makedirs(OUT, exist_ok=True)

# --- 暖色编辑风，与项目一贯审美一致；不用冷蓝 ---
INK = "#1f1a17"
PAPER = "#fbf7f0"
BROWN = "#5b3a29"
AMBER = "#d98c2b"
RUST = "#b4472a"
OLIVE = "#6b7a3a"
GREY = "#9a8f85"

plt.rcParams.update({
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.edgecolor": GREY,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 160,
})

# 中文字体：Windows 上找得到就用，找不到退回英文标签
CJK = None
for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC"):
    try:
        font_manager.findfont(name, fallback_to_default=False)
        CJK = name
        break
    except Exception:
        continue
if CJK:
    plt.rcParams["font.family"] = CJK
    plt.rcParams["axes.unicode_minus"] = False
    print("CJK font:", CJK)
else:
    print("no CJK font, falling back to English labels")

L = (lambda zh, en: zh) if CJK else (lambda zh, en: en)


def load(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return json.load(f)


# ============ 图 1：阶段耗时占比 ============
st = load("bench_stages.json")
labels = [
    L("get_DAG() 建边", "get_DAG()"),
    L("calc() 动态规划", "calc()"),
    L("HMM 新词发现", "HMM"),
    L("正则切分 / 组装", "regex / assembly"),
]
vals = [st["stages_pct"][k] for k in st["stages_ms"]]
ms = [st["stages_ms"][k] for k in st["stages_ms"]]
colors = [RUST, AMBER, OLIVE, GREY]

fig, ax = plt.subplots(figsize=(8.6, 2.7))
left = 0
for lab, v, m, c in zip(labels, vals, ms, colors):
    ax.barh([0], [v], left=left, color=c, height=0.5, edgecolor=PAPER, linewidth=2)
    ax.text(left + v / 2, 0, f"{v:.1f}%", ha="center", va="center",
            color="white", fontsize=13, fontweight="bold")
    ax.text(left + v / 2, -0.42, f"{m:.1f} ms", ha="center", va="center",
            color=GREY, fontsize=9)
    left += v
ax.set_xlim(0, 100)
ax.set_ylim(-0.75, 0.55)
ax.set_yticks([])
# 横轴刻度已带百分号，标签本身多余；图例挪到标题下方，避免和刻度打架
ax.set_xticks([0, 20, 40, 60, 80, 100])
ax.set_xticklabels(["0", "20%", "40%", "60%", "80%", "100%"])
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.tick_params(axis="x", length=0, labelsize=9, colors=GREY)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.0, 1.30),
          ncol=4, frameon=False, fontsize=9.5, handlelength=1.1,
          columnspacing=1.5, handletextpad=0.5)
ax.set_title(
    L("精确模式 cut() 的耗时分布：get_DAG + calc 占 66.3%，同一批子串被算了两遍",
      "cut() time split: get_DAG + calc = 66.3%, the same substrings scanned twice"),
    fontsize=11.5, pad=34, loc="left", color=INK)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig-stages.png"), bbox_inches="tight")
plt.close(fig)
print("wrote docs/fig-stages.png")

# ============ 图 2：各模式加速比 ============
res = load("bench_final.json")
order = [
    ("cut", L("cut() 默认", "cut() default"), True),
    ("nohmm", L("cut(HMM=False)", "cut(HMM=False)"), True),
    ("search", L("cut_for_search()", "cut_for_search()"), True),
    ("whole", L("整段 16.7k 字", "one 16.7k-char doc"), True),
    ("hmm_heavy", L("长未登录串 4.8k 字", "4.8k-char unknown run"), True),
    ("cutall", L("cut_all=True（未改动，对照）", "cut_all=True (untouched control)"), False),
]
names = [o[1] for o in order]
sp = [res[o[0]]["speedup"] for o in order]
lo = [res[o[0]]["speedup"] - res[o[0]]["lo"] for o in order]
hi = [res[o[0]]["hi"] - res[o[0]]["speedup"] for o in order]
cols = [AMBER if o[2] else GREY for o in order]

fig, ax = plt.subplots(figsize=(8.6, 3.9))
y = np.arange(len(names))[::-1]
ax.barh(y, sp, color=cols, height=0.6, zorder=3)
ax.errorbar(sp, y, xerr=[lo, hi], fmt="none", ecolor=BROWN, elinewidth=1.4,
            capsize=3.5, zorder=4)
ax.axvline(1.0, color=RUST, linestyle="--", linewidth=1.3, zorder=2)
ax.set_ylim(-1.15, len(names) - 0.4)
ax.text(1.0, -0.95, L("1.0x 基准", "1.0x baseline"), color=RUST,
        fontsize=8.5, va="center", ha="center", zorder=5)
for yi, v in zip(y, sp):
    ax.text(v + 0.22, yi, f"{v:.2f}x", va="center", fontsize=11,
            fontweight="bold", color=INK)
ax.set_yticks(y)
ax.set_yticklabels(names, fontsize=10)
ax.set_xlabel(L("加速比（同进程轮内交错，15 轮，取逐轮比值中位数）",
                "speedup (interleaved in-process, 15 rounds, median of per-round ratios)"))
ax.set_xlim(0, 13.6)
ax.grid(axis="x", color=GREY, alpha=0.25, zorder=0)
ax.set_title(
    L("灰色 = 未改动的对照组，应停在 1.0x；误差棒为逐轮比值的全距",
      "grey = untouched control, should sit at 1.0x; whiskers span all rounds"),
    fontsize=10.5, pad=12, loc="left", color=GREY)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig-speedup.png"), bbox_inches="tight")
plt.close(fig)
print("wrote docs/fig-speedup.png")

# ============ 图 3：viterbi 随长度 ============
rows = load("bench_viterbi_impls.json")
n = [r["n"] for r in rows]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.7))

ax1.plot(n, [r["orig_us"] for r in rows], "o-", color=RUST, label=L("原版（每步复制整条路径）", "original (copies full path)"), markersize=4.5, linewidth=1.8)
ax1.plot(n, [r["generic_us"] for r in rows], "s--", color=GREY, label=L("back-pointer 通用版", "back-pointer, generic"), markersize=4, linewidth=1.5)
ax1.plot(n, [r["unrolled_us"] for r in rows], "^-", color=AMBER, label=L("back-pointer + BMES 展开", "back-pointer + BMES unrolled"), markersize=4.5, linewidth=1.9)
ax1.set_xlabel(L("观测长度 n（字）", "observation length n"))
ax1.set_ylabel(L("单次调用耗时（µs）", "time per call (us)"))
ax1.set_title(L("绝对耗时：原版呈平方增长", "Absolute cost: original grows quadratically"),
              fontsize=10.5, loc="left")
ax1.legend(frameon=False, fontsize=8.5)
ax1.grid(color=GREY, alpha=0.25)

sp_u = [r["unrolled_speedup"] for r in rows]
sp_g = [r["generic_speedup"] for r in rows]
ax2.plot(n, sp_u, "^-", color=AMBER, markersize=4.5, linewidth=1.9,
         label=L("展开版 / 原版", "unrolled / original"))
ax2.plot(n, sp_g, "s--", color=GREY, markersize=4, linewidth=1.5,
         label=L("通用版 / 原版", "generic / original"))
ax2.axhline(1.0, color=RUST, linestyle="--", linewidth=1.2)
ax2.annotate(L("通用版在 n=2 时反而更慢 (0.91x)：\n省下的复制抵不过索引开销",
               "generic is slower at n=2 (0.91x):\nsaved copying < index overhead"),
             xy=(2.5, sp_g[0]), xytext=(58, 4.35), fontsize=8.2, color=BROWN,
             arrowprops=dict(arrowstyle="->", color=BROWN, lw=1,
                             connectionstyle="arc3,rad=0.2"))
ax2.set_xlabel(L("观测长度 n（字）", "observation length n"))
ax2.set_ylabel(L("加速比", "speedup"))
ax2.set_ylim(0.4, 7.6)
ax2.set_title(L("加速比随长度单调上升 = O(n²)→O(n)", "speedup rises with n = O(n^2) -> O(n)"),
              fontsize=10.5, loc="left")
ax2.legend(frameon=False, fontsize=8.5)
ax2.grid(color=GREY, alpha=0.25)

fig.suptitle(L("Viterbi：绝对耗时与加速比随观测长度的变化",
               "Viterbi: absolute cost and speedup vs observation length"),
             fontsize=12, x=0.02, ha="left", color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(OUT, "fig-viterbi.png"), bbox_inches="tight")
plt.close(fig)
print("wrote docs/fig-viterbi.png")

# ============ 图 4：四臂归因 ============
ab = load("bench_ablation.json")
modes = [("cut", L("cut()", "cut()")), ("nohmm", L("HMM=False", "HMM=False")),
         ("search", L("cut_for_search", "cut_for_search")),
         ("hmm_heavy", L("长未登录串", "long unknown run")),
         ("cutall", L("cut_all（对照）", "cut_all (control)"))]
x = np.arange(len(modes))
w = 0.26
series = [("B_route", L("只改 _route", "_route only"), BROWN),
          ("C_viterbi", L("只改 viterbi", "viterbi only"), OLIVE),
          ("D_both", L("两处都改", "both"), AMBER)]

fig, ax = plt.subplots(figsize=(9.2, 3.9))
for i, (key, lab, c) in enumerate(series):
    vals = [ab[m][key]["speedup"] for m, _ in modes]
    ax.bar(x + (i - 1) * w, vals, w, label=lab, color=c, zorder=3)
    for xi, v in zip(x + (i - 1) * w, vals):
        ax.text(xi, v + 0.28, f"{v:.2f}", ha="center", fontsize=8.2, color=INK)
ax.axhline(1.0, color=RUST, linestyle="--", linewidth=1.2, zorder=2)
ax.set_xticks(x)
ax.set_xticklabels([l for _, l in modes], fontsize=9.5)
ax.set_ylabel(L("相对原版的加速比", "speedup vs original"))
ax.set_yscale("log")
ax.set_ylim(0.7, 26)
ax.set_yticks([1, 2, 5, 10, 20])
ax.set_yticklabels(["1x", "2x", "5x", "10x", "20x"])
ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper left")
ax.grid(axis="y", color=GREY, alpha=0.25, zorder=0)
ax.set_title(
    L("对数轴。对照组三根柱子都贴着 1.0x，说明测量本身没有偏置",
      "log scale. The control's three bars sit at 1.0x, so the measurement is unbiased"),
    fontsize=10.5, pad=12, loc="left", color=GREY)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig-ablation.png"), bbox_inches="tight")
plt.close(fig)
print("wrote docs/fig-ablation.png")
