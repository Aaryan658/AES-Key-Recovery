"""Download the ASCAD fixed-key (ATMEGA_AES_v1) dataset and extract ASCAD.h5.

The ANSSI-FR/ASCAD project distributes the fixed-key campaign as a single
~4.4 GB archive ``ASCAD_data.zip`` hosted on data.gouv.fr. Inside it, the file
we actually need for a first-round S-box attack is::

    ASCAD_data/ASCAD_databases/ASCAD.h5      (~250 MB)

which holds 50,000 profiling + 10,000 attack traces of 700 samples each,
already windowed around the masked S-box operation of key byte 2.

This script downloads the archive (resumable), extracts only ``ASCAD.h5``
(and, optionally, the desync variants), verifies it opens, and prints its
structure. Pass ``--h5-url`` if you have a faster direct mirror of ASCAD.h5,
or ``--zip-path`` / ``--from-h5`` if you already downloaded something.

Usage
-----
    python -m data.download_ascad                     # full download + extract
    python -m data.download_ascad --with-desync       # also extract desync50/100
    python -m data.download_ascad --zip-path X.zip    # use an existing archive
    python -m data.download_ascad --h5-url  <URL>     # direct ASCAD.h5 mirror
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# Official archive (fixed-key campaign). Verified against
# https://github.com/ANSSI-FR/ASCAD/.../ATM_AES_v1_fixed_key/Readme.md
ASCAD_ZIP_URL = "https://static.data.gouv.fr/resources/ascad/20180530-163000/ASCAD_data.zip"
ASCAD_ZIP_BYTES = 4_435_199_469  # for a sanity check / progress bar total

# ATMEGA_AES_v1 variable-key extracted database: a single ~418 MB .h5, no zip.
# 200000 profiling + 100000 attack traces, 1400 samples. Milestone 2 target.
ASCAD_VARIABLE_H5_URL = "https://www.data.gouv.fr/api/1/datasets/r/b4ace767-c2a4-4db4-8e01-4527b5b91f00"
ASCAD_VARIABLE_NAME = "ascad-variable.h5"

# Members we may want out of the archive. Names are matched by suffix so a
# leading "ASCAD_data/ASCAD_databases/" (or any prefix) is tolerated.
CORE_MEMBER = "ASCAD.h5"
DESYNC_MEMBERS = ("ASCAD_desync50.h5", "ASCAD_desync100.h5")

DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "raw"
CHUNK = 1 << 20  # 1 MiB


def _stream_download(url: str, dest: Path, expected_bytes: "int | None" = None) -> None:
    """Download ``url`` to ``dest`` with an HTTP-range resume and a progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0

    headers = {}
    if have:
        headers["Range"] = f"bytes={have}-"
        print(f"  resuming from {have / 1e9:.2f} GB")

    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        if r.status_code == 416:  # range not satisfiable -> already complete
            print("  file already fully downloaded")
            return
        r.raise_for_status()
        total = expected_bytes
        if r.headers.get("Content-Range"):
            total = int(r.headers["Content-Range"].split("/")[-1])
        elif r.headers.get("Content-Length"):
            total = have + int(r.headers["Content-Length"])

        mode = "ab" if have else "wb"
        with open(dest, mode) as fh, tqdm(
            total=total, initial=have, unit="B", unit_scale=True, unit_divisor=1024,
            desc=f"  {dest.name}",
        ) as bar:
            for chunk in r.iter_content(chunk_size=CHUNK):
                if chunk:
                    fh.write(chunk)
                    bar.update(len(chunk))


def _extract_members(zip_path: Path, out_dir: Path, wanted: list) -> list:
    """Extract archive members whose name ends with one of ``wanted``. Flattens paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        for target in wanted:
            match = next((n for n in names if n.replace("\\", "/").endswith(target)), None)
            if match is None:
                print(f"  !! {target} not found in archive", file=sys.stderr)
                continue
            dest = out_dir / target
            if dest.exists():
                print(f"  {target} already extracted -> {dest}")
                written.append(dest)
                continue
            info = zf.getinfo(match)
            with zf.open(match) as src, open(dest, "wb") as dst, tqdm(
                total=info.file_size, unit="B", unit_scale=True, unit_divisor=1024,
                desc=f"  extract {target}",
            ) as bar:
                while True:
                    buf = src.read(CHUNK)
                    if not buf:
                        break
                    dst.write(buf)
                    bar.update(len(buf))
            written.append(dest)
    return written


def _verify_h5(path: Path) -> None:
    """Open the HDF5 file and print its structure so problems surface early."""
    import h5py

    print(f"\nVerifying {path} ({path.stat().st_size / 1e6:.1f} MB)")
    with h5py.File(path, "r") as f:
        for grp in ("Profiling_traces", "Attack_traces"):
            if grp not in f:
                raise RuntimeError(f"expected group {grp!r} missing from {path}")
            g = f[grp]
            traces = g["traces"]
            labels = g["labels"]
            meta = g["metadata"]
            print(f"  /{grp}/traces   shape={traces.shape} dtype={traces.dtype}")
            print(f"  /{grp}/labels   shape={labels.shape} dtype={labels.dtype}")
            print(f"  /{grp}/metadata fields={meta.dtype.names}")
    print("OK - ASCAD.h5 looks valid.\n")


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                    help=f"where to put the .h5 (default: {DEFAULT_OUT_DIR})")
    ap.add_argument("--name", default=CORE_MEMBER,
                    help=f"output filename (default: {CORE_MEMBER})")
    ap.add_argument("--variable", action="store_true",
                    help="download the ATMEGA_AES_v1 variable-key set (ascad-variable.h5, ~418 MB)")
    ap.add_argument("--zip-path", type=Path, default=None,
                    help="use this already-downloaded ASCAD_data.zip instead of downloading")
    ap.add_argument("--zip-url", default=ASCAD_ZIP_URL, help="override the archive URL")
    ap.add_argument("--h5-url", default=None,
                    help="direct URL to a standalone .h5 (skips the 4.4 GB archive)")
    ap.add_argument("--from-h5", type=Path, default=None,
                    help="copy/verify an .h5 you already have on disk")
    ap.add_argument("--with-desync", action="store_true",
                    help="also extract ASCAD_desync50.h5 and ASCAD_desync100.h5")
    ap.add_argument("--keep-zip", action="store_true", help="do not delete the archive afterwards")
    args = ap.parse_args(argv)

    if args.variable:
        # convenience preset for the variable-key single-file database
        if args.h5_url is None:
            args.h5_url = ASCAD_VARIABLE_H5_URL
        if args.name == CORE_MEMBER:
            args.name = ASCAD_VARIABLE_NAME

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    core_dest = out_dir / args.name

    if core_dest.exists():
        print(f"{core_dest} already present - verifying only.")
        _verify_h5(core_dest)
        return 0

    # Path A: user already has the .h5
    if args.from_h5 is not None:
        src = args.from_h5.resolve()
        if not src.exists():
            ap.error(f"--from-h5 {src} does not exist")
        if src != core_dest:
            import shutil
            shutil.copy2(src, core_dest)
        _verify_h5(core_dest)
        return 0

    # Path B: direct standalone ASCAD.h5 mirror
    if args.h5_url:
        print(f"Downloading standalone ASCAD.h5 from {args.h5_url}")
        _stream_download(args.h5_url, core_dest)
        _verify_h5(core_dest)
        return 0

    # Path C: the official archive (download unless one was supplied)
    zip_path = args.zip_path
    if zip_path is None:
        zip_path = out_dir / "ASCAD_data.zip"
        print(f"Downloading ASCAD archive (~4.4 GB) from {args.zip_url}")
        print("This is large; the download is resumable - rerun if it drops.")
        _stream_download(args.zip_url, zip_path, expected_bytes=ASCAD_ZIP_BYTES)
    elif not zip_path.exists():
        ap.error(f"--zip-path {zip_path} does not exist")

    wanted = [CORE_MEMBER] + (list(DESYNC_MEMBERS) if args.with_desync else [])
    print(f"\nExtracting {wanted} from {zip_path}")
    _extract_members(zip_path, out_dir, wanted)
    _verify_h5(core_dest)

    if zip_path == out_dir / "ASCAD_data.zip" and not args.keep_zip:
        print(f"Removing archive {zip_path} (pass --keep-zip to retain it).")
        os.remove(zip_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
