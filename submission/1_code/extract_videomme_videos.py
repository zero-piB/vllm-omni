#!/usr/bin/env python3
"""Extract VideoMME videos needed by the eval parquet from the 20 chunked zips.

Reads /workspace/shared_assets/datasets/lmms-lab/Video-MME/videomme/test-00000-of-00001.parquet
to get the set of videoIDs, then extracts only those mp4 files from the 20
videos_chunked_*.zip archives into /workspace/submission/7_runtime/media/videomme_videos/{videoID}.mp4.

Usage:
  python3 extract_videomme_videos.py [--parquet PATH] [--zips-dir PATH] [--out-dir PATH]
"""
import argparse, json, os, sys, zipfile
from pathlib import Path


def load_video_ids(parquet: Path) -> set[str]:
    import pyarrow.parquet as pq
    ids: set[str] = set()
    # Try both possible id columns; filename key is videoID per the dataset README.
    for col in ("videoID", "video_id"):
        try:
            t = pq.read_table(parquet, columns=[col])
        except Exception:
            continue
        for v in t.column(col).to_pylist():
            if v:
                ids.add(str(v))
    if not ids:
        raise SystemExit(f"No video IDs found in {parquet}")
    print(f"[load] {len(ids)} video IDs from {parquet}", file=sys.stderr)
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    raw = os.environ.get("RAW_DATA_DIR", "/workspace/shared_assets/datasets")
    sub = os.environ.get("SUBMISSION_DIR", "/workspace/submission")
    ap.add_argument("--parquet", type=Path,
                    default=Path(raw) / "lmms-lab/Video-MME/videomme/test-00000-of-00001.parquet")
    ap.add_argument("--zips-dir", type=Path,
                    default=Path(raw) / "lmms-lab/Video-MME")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(sub) / "7_runtime/media/videomme_videos")
    ap.add_argument("--zip-prefix", default="videos_chunked")
    args = ap.parse_args()

    ids = load_video_ids(args.parquet)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    zips = sorted(args.zips_dir.glob(f"{args.zip_prefix}_*.zip"))
    if not zips:
        raise SystemExit(f"No zips matching {args.zip_prefix}_*.zip under {args.zips_dir}")

    # Build videoID -> zip + member name index from central directories.
    index: dict[str, tuple[Path, str]] = {}
    for zp in zips:
        with zipfile.ZipFile(zp) as z:
            for info in z.infolist():
                name = Path(info.filename).name  # data/{videoID}.mp4
                if name.endswith(".mp4"):
                    key = name[:-4]
                    if key in ids:
                        index.setdefault(key, (zp, info.filename))
    print(f"[index] {len(index)} needed videos located in {len(zips)} zips", file=sys.stderr)
    missing = ids - set(index)
    if missing:
        print(f"[warn] {len(missing)} ids not found in zips, e.g. {sorted(missing)[:5]}", file=sys.stderr)

    # Extract.
    done = 0
    for key, (zp, member) in sorted(index.items()):
        out = args.out_dir / f"{key}.mp4"
        if out.exists() and out.stat().st_size > 0:
            done += 1
            continue
        try:
            with zipfile.ZipFile(zp) as z:
                with z.open(member) as src, open(out, "wb") as dst:
                    while True:
                        chunk = src.read(1 << 20)
                        if not chunk:
                            break
                        dst.write(chunk)
            done += 1
        except Exception as e:
            print(f"[error] {key}: {e}", file=sys.stderr)
        if done % 50 == 0:
            print(f"[progress] {done}/{len(index)}", file=sys.stderr, flush=True)

    print(f"[done] extracted {done}/{len(index)} videos to {args.out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
