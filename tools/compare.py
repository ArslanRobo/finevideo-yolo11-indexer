"""Score a results file against a reference file, with the brief's metrics.

    python tools/compare.py results.jsonl reference.jsonl

Section 5 of the brief:
  * objects F1 >= 0.95       (over every (video, second, object) triple)
  * people exact  >= 85%     (of the reference's seconds)
  * people within 1 >= 95%

Only videos present in the REFERENCE are scored, so a reference built on a
subset of files can be compared with a full results file. A second the
results file is missing counts as wrong (no objects, no people match).
"""

import argparse
import gzip
import json
import sys
from collections import Counter

THRESHOLDS = {"objects_f1": 0.95, "people_exact": 0.85, "people_within_1": 0.95}


def load(path):
    """Return ({(video, second): (set_of_objects, people)}, {video: error})."""
    opener = gzip.open if str(path).endswith(".gz") else open
    seconds, errors = {}, {}
    with opener(path, "rt") as f:
        for line in f:
            row = json.loads(line)
            if "error" in row:
                errors[row["video"]] = row["error"]
            else:
                seconds[(row["video"], row["second"])] = (set(row["objects"]), row["people"])
    return seconds, errors


def score(results, reference):
    ref_videos = {video for video, _ in reference}
    tp = fp = fn = 0
    exact = within_1 = 0
    per_object = Counter()  # (object, "tp"/"fp"/"fn") -> count

    for key, (ref_objects, ref_people) in reference.items():
        got_objects, got_people = results.get(key, (set(), None))
        for obj in got_objects & ref_objects:
            per_object[obj, "tp"] += 1
        for obj in got_objects - ref_objects:
            per_object[obj, "fp"] += 1
        for obj in ref_objects - got_objects:
            per_object[obj, "fn"] += 1
        tp += len(got_objects & ref_objects)
        fp += len(got_objects - ref_objects)
        fn += len(ref_objects - got_objects)
        if got_people is not None:
            exact += got_people == ref_people
            within_1 += abs(got_people - ref_people) <= 1

    # Seconds the results claim that the reference does not have.
    extra = [k for k in results if k[0] in ref_videos and k not in reference]
    for key in extra:
        fp += len(results[key][0])

    n = len(reference)
    f1 = 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else 1.0
    return {
        "seconds": n,
        "objects_f1": f1,
        "people_exact": exact / n if n else 1.0,
        "people_within_1": within_1 / n if n else 1.0,
        "missing_seconds": sum(1 for k in reference if k not in results),
        "extra_seconds": len(extra),
        "per_object": per_object,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results")
    parser.add_argument("reference")
    args = parser.parse_args()

    results, result_errors = load(args.results)
    reference, reference_errors = load(args.reference)
    s = score(results, reference)

    print(f"seconds scored: {s['seconds']}  (missing {s['missing_seconds']}, extra {s['extra_seconds']})")
    passed = True
    for metric, threshold in THRESHOLDS.items():
        ok = s[metric] >= threshold
        passed &= ok
        print(f"  {metric:16s} {s[metric]:.4f}   (need >= {threshold})  {'PASS' if ok else 'FAIL'}")

    print("  per object:")
    objects = sorted({obj for obj, _ in s["per_object"]})
    for obj in objects:
        tp, fp, fn = (s["per_object"][obj, k] for k in ("tp", "fp", "fn"))
        f1 = 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else 1.0
        print(f"    {obj:11s} F1 {f1:.4f}  (tp {tp}, fp {fp}, fn {fn})")

    ref_videos = {v for v, _ in reference} | set(reference_errors)
    error_mismatch = {v for v in ref_videos if (v in result_errors) != (v in reference_errors)}
    print(f"  error lines: results {len(result_errors)}, reference {len(reference_errors)}, "
          f"disagree on {len(error_mismatch)} video(s)")
    print("OVERALL:", "PASS" if passed else "FAIL")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
