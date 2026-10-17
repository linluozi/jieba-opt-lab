from __future__ import absolute_import, unicode_literals
import re
import os
import sys
import pickle
from .._compat import *

MIN_FLOAT = -3.14e100

PROB_START_P = "prob_start.p"
PROB_TRANS_P = "prob_trans.p"
PROB_EMIT_P = "prob_emit.p"


PrevStatus = {
    'B': 'ES',
    'M': 'MB',
    'S': 'SE',
    'E': 'BM'
}

Force_Split_Words = set([])
def load_model():
    start_p = pickle.load(get_module_res("finalseg", PROB_START_P))
    trans_p = pickle.load(get_module_res("finalseg", PROB_TRANS_P))
    emit_p = pickle.load(get_module_res("finalseg", PROB_EMIT_P))
    return start_p, trans_p, emit_p

if sys.platform.startswith("java"):
    start_P, trans_P, emit_P = load_model()
else:
    from .prob_start import P as start_P
    from .prob_trans import P as trans_P
    from .prob_emit import P as emit_P


# --- Pre-resolved constants for the unrolled BMES path (see _viterbi_bmes) ---
# Resolved once at import time so the per-character loop does no dict lookups
# for the transition matrix or the start probabilities.
_BMES = ('B', 'M', 'E', 'S')
_START_P = {s: start_P[s] for s in _BMES}
_EMIT_B, _EMIT_M, _EMIT_E, _EMIT_S = emit_P['B'], emit_P['M'], emit_P['E'], emit_P['S']
# _TR_<from><to>: log P(next tag = to | current tag = from)
_TR_BM = trans_P['B']['M']
_TR_BE = trans_P['B']['E']
_TR_MM = trans_P['M']['M']
_TR_ME = trans_P['M']['E']
_TR_EB = trans_P['E']['B']
_TR_ES = trans_P['E']['S']
_TR_SB = trans_P['S']['B']
_TR_SS = trans_P['S']['S']


def viterbi(obs, states, start_p, trans_p, emit_p):
    """Viterbi over the BMES tag chain (optimization, see PATCH.md).

    The original kept a full tag *path* per state and copied it at every step
    (``path[state] + [y]``). For a chain of length n that copies
    n + (n-1) + ... + 1 = O(n^2) list elements in total, although only one
    path is ever returned. Here each step stores just the chosen predecessor
    of every state (a back-pointer) and the path is rebuilt once at the end:
    O(n) memory, O(n) copying.

    Results are identical, including tie-breaking. The original picks with
    ``max((prob, y0) for y0 in PrevStatus[y])``, i.e. on an exact tie it keeps
    the y0 whose *character* is larger ('S' > 'E', 'M' > 'B'). The comparison
    below reproduces that rule explicitly instead of relying on tuple ordering.

    ``states`` is always 'BMES' in this package, and callers are dominated by
    2-4 character inputs, so that case takes a fully unrolled path with the
    transition matrix pre-resolved to constants; the generic loop remains for
    any other state set.
    """
    if states == 'BMES' and start_p is start_P and trans_p is trans_P and emit_p is emit_P:
        return _viterbi_bmes(obs)
    return _viterbi_generic(obs, states, start_p, trans_p, emit_p)


def _viterbi_bmes(obs):
    """Fully unrolled Viterbi for the built-in BMES model.

    PrevStatus has exactly two predecessors per state, so the inner loop is
    unrolled and the transition matrix is resolved once at import time. The
    tie-break order (the larger state character wins) is spelled out in each
    branch: B<-S,E  M<-M,B  E<-M,B  S<-S,E.
    """
    n = len(obs)
    cur_b = _START_P['B'] + _EMIT_B.get(obs[0], MIN_FLOAT)
    cur_m = _START_P['M'] + _EMIT_M.get(obs[0], MIN_FLOAT)
    cur_e = _START_P['E'] + _EMIT_E.get(obs[0], MIN_FLOAT)
    cur_s = _START_P['S'] + _EMIT_S.get(obs[0], MIN_FLOAT)

    # back[t] packs the four chosen predecessors as a 4-tuple of small ints.
    back = []

    for t in xrange(1, n):
        ch = obs[t]
        e_b = _EMIT_B.get(ch, MIN_FLOAT)
        e_m = _EMIT_M.get(ch, MIN_FLOAT)
        e_e = _EMIT_E.get(ch, MIN_FLOAT)
        e_s = _EMIT_S.get(ch, MIN_FLOAT)

        # B <- {S, E}; on a tie 'S' (larger char) wins.
        v = cur_s + _TR_SB + e_b
        w = cur_e + _TR_EB + e_b
        if v >= w:
            n_b, p_b = v, 3
        else:
            n_b, p_b = w, 2

        # M <- {M, B}; on a tie 'M' wins.
        v = cur_m + _TR_MM + e_m
        w = cur_b + _TR_BM + e_m
        if v >= w:
            n_m, p_m = v, 1
        else:
            n_m, p_m = w, 0

        # E <- {B, M}; on a tie 'M' wins.
        v = cur_b + _TR_BE + e_e
        w = cur_m + _TR_ME + e_e
        if v > w:
            n_e, p_e = v, 0
        else:
            n_e, p_e = w, 1

        # S <- {S, E}; on a tie 'S' wins.
        v = cur_s + _TR_SS + e_s
        w = cur_e + _TR_ES + e_s
        if v >= w:
            n_s, p_s = v, 3
        else:
            n_s, p_s = w, 2

        back.append((p_b, p_m, p_e, p_s))
        cur_b, cur_m, cur_e, cur_s = n_b, n_m, n_e, n_s

    # max((prob, y) for y in 'ES'); on a tie 'S' wins.
    if cur_e > cur_s:
        prob, state = cur_e, 2
    else:
        prob, state = cur_s, 3

    path = [0] * n
    path[n - 1] = state
    for t in xrange(n - 2, -1, -1):
        state = back[t][state]
        path[t] = state

    return (prob, [_BMES[i] for i in path])


def _viterbi_generic(obs, states, start_p, trans_p, emit_p):
    """Generic Viterbi: same results, any state set."""
    n = len(obs)
    order = list(states)
    index = {s: i for i, s in enumerate(order)}
    # Pre-resolve PrevStatus into index lists, keeping the original iteration
    # order so that tie-breaking stays identical.
    preds = [[index[s] for s in PrevStatus[y]] for y in order]
    # trans[i][j] == trans_p[order[i]].get(order[j], MIN_FLOAT)
    trans = [[trans_p[a].get(b, MIN_FLOAT) for b in order] for a in order]

    cur = [start_p[y] + emit_p[y].get(obs[0], MIN_FLOAT) for y in order]
    back = []

    for t in xrange(1, n):
        ch = obs[t]
        nxt = [0.0] * len(order)
        step = [0] * len(order)
        for j in xrange(len(order)):
            em_p = emit_p[order[j]].get(ch, MIN_FLOAT)
            best = None
            best_i = 0
            best_char = None
            for i0 in preds[j]:
                v = cur[i0] + trans[i0][j] + em_p
                c = order[i0]
                if best is None or v > best or (v == best and c > best_char):
                    best = v
                    best_i = i0
                    best_char = c
            nxt[j] = best
            step[j] = best_i
        back.append(step)
        cur = nxt

    # max((prob, y) for y in 'ES'): on a tie 'S' wins because 'S' > 'E'.
    e_i, s_i = index['E'], index['S']
    if cur[e_i] > cur[s_i]:
        prob, state_i = cur[e_i], e_i
    else:
        prob, state_i = cur[s_i], s_i

    path = [order[state_i]]
    for t in xrange(n - 2, -1, -1):
        state_i = back[t][state_i]
        path.append(order[state_i])
    path.reverse()

    return (prob, path)


def __cut(sentence):
    global emit_P
    prob, pos_list = viterbi(sentence, 'BMES', start_P, trans_P, emit_P)
    begin, nexti = 0, 0
    # print pos_list, sentence
    for i, char in enumerate(sentence):
        pos = pos_list[i]
        if pos == 'B':
            begin = i
        elif pos == 'E':
            yield sentence[begin:i + 1]
            nexti = i + 1
        elif pos == 'S':
            yield char
            nexti = i + 1
    if nexti < len(sentence):
        yield sentence[nexti:]

re_han = re.compile("([\u4E00-\u9FD5]+)")
re_skip = re.compile("([a-zA-Z0-9]+(?:\.\d+)?%?)")


def add_force_split(word):
    global Force_Split_Words
    Force_Split_Words.add(word)

def cut(sentence):
    sentence = strdecode(sentence)
    blocks = re_han.split(sentence)
    for blk in blocks:
        if re_han.match(blk):
            for word in __cut(blk):
                if word not in Force_Split_Words:
                    yield word
                else:
                    for c in word:
                        yield c
        else:
            tmp = re_skip.split(blk)
            for x in tmp:
                if x:
                    yield x
