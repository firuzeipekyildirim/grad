#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from sync_report_assets import sync_outputs_to_report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="vrpoutputs")
    ap.add_argument("--report_dir", default="rapor")
    ap.add_argument("--xlsx_to_csv", action="store_true")
    ap.add_argument("--drop_excel", action="store_true")
    ap.add_argument("--patterns", nargs="*", default=["*"])
    args = ap.parse_args()

    created = sync_outputs_to_report(
        outdir=Path(args.outdir),
        report_dir=Path(args.report_dir),
        excel_split_to_csv=bool(args.xlsx_to_csv),
        keep_excel=(not args.drop_excel),
        patterns=tuple(args.patterns),
    )

    print(f"[OK] synced_to_report: {len(created)} item(s)")
    for p in created[:20]:
        print(" -", p)

if __name__ == "__main__":
    main()
