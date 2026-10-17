"""准备对照组：从 optimized/ 反向打补丁，逐字节重建上游原始代码。

这样压缩包里只需要带一份源码（55 MB 的词典/模型不用带两遍），
同时保证对照组不是"我手抄的"，而是补丁的逆运算结果 —— 可用 SHA256 核对。
"""
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.join(HERE, "original")
OPT = os.path.join(HERE, "optimized")
PATCH = os.path.join(HERE, "jieba-optimization.patch")

EXPECTED = {
    "jieba/__init__.py":
        "b47cefcfd0bf683b0c708c1de0328df83be2c35264637c442106e79d739fb7f3",
    "jieba/finalseg/__init__.py":
        "2168475bf522c0dae9d0a5e2bcb63c11b8c15931a622035cb8905d1e77e94f84",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if os.path.isdir(ORIG):
        ok = all(sha256(os.path.join(ORIG, k)) == v for k, v in EXPECTED.items())
        if ok:
            print("original/ already present and matches upstream SHA256.")
            return 0
        print("original/ exists but does not match; rebuilding.")
        shutil.rmtree(ORIG)

    print("Rebuilding original/ by reverse-applying the patch ...")

    # Apply in a temp dir OUTSIDE any git repository, then move it into place.
    # If we ran `git apply` with cwd=<project>/original, git would notice that
    # original/ sits inside the project's own repository, resolve -p1 paths
    # against that repo root, and quietly patch optimized/ instead -- leaving
    # original/ untouched and reporting success. Keeping the work dir outside
    # any repo makes the path resolution unambiguous.
    work = tempfile.mkdtemp(prefix="jieba-recon-")
    try:
        stage = os.path.join(work, "tree")
        shutil.copytree(OPT, stage)
        shutil.rmtree(os.path.join(stage, ".git"), ignore_errors=True)

        cmd = ["git", "-c", "core.autocrlf=false", "apply", "-R", "-p1",
               "--whitespace=nowarn", os.path.abspath(PATCH)]
        r = subprocess.run(cmd, cwd=stage, capture_output=True, text=True)
        if r.returncode != 0:
            print("git apply failed:", r.stderr)
            return 1

        bad = 0
        for rel, want in EXPECTED.items():
            got = sha256(os.path.join(stage, rel))
            flag = "OK" if got == want else "MISMATCH"
            if got != want:
                bad += 1
            print(f"  {rel:32s} {flag}")
        if bad:
            print("\nRebuild does not match upstream. Aborting.")
            return 1

        shutil.move(stage, ORIG)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print("\noriginal/ rebuilt and verified against upstream SHA256.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
