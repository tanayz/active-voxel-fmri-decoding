"""Merge per-subject haxby_vt_final_*.json files (from split Kaggle/local runs) into one results file.

Usage: python merge_haxby_final.py OUT.json IN1.json IN2.json ...
Checks that every input used the same config and the same library versions; refuses to merge otherwise
(a mixed-version merge would silently invalidate the 'all numbers from one procedure' claim).
"""
import json
import sys


def main():
    out, ins = sys.argv[1], sys.argv[2:]
    merged, ref_cfg, ref_ver = None, None, None
    for path in ins:
        r = json.load(open(path))
        cfg = {k: v for k, v in r["config"].items()}
        ver = {k: r["versions"][k] for k in ("scikit-learn", "nilearn", "numpy")}
        if merged is None:
            merged, ref_cfg, ref_ver = {"versions": r["versions"], "config": r["config"], "subjects": {},
                                        "sources": {}}, cfg, ver
        if cfg != ref_cfg:
            sys.exit(f"config mismatch in {path}:\n  {cfg}\n  vs\n  {ref_cfg}")
        if ver != ref_ver:
            sys.exit(f"library-version mismatch in {path}: {ver} vs {ref_ver}")
        for s, v in r["subjects"].items():
            if s in merged["subjects"]:
                print(f"note: subject {s} already present, keeping first occurrence; skipping {path}")
                continue
            merged["subjects"][s] = v
            merged["sources"][s] = path
    json.dump(merged, open(out, "w"), indent=2)
    subs = sorted(merged["subjects"], key=int)
    print(f"merged subjects {subs} -> {out}")
    for s in subs:
        r = merged["subjects"][s]
        print(f"  S{s}: ours={r['active_voxel']['mean']:.3f}±{r['active_voxel']['sd']:.3f} "
              f"base={r['all_vt_baseline']['mean']:.3f}±{r['all_vt_baseline']['sd']:.3f} "
              f"p={r['permutation']['p_value']:.4f} null={r['permutation']['null_mean']:.3f} "
              f"(max {r['permutation']['null_max']:.3f}) k={r['active_voxel']['k_histogram']}")


if __name__ == "__main__":
    main()
