"""Command line interface."""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .stack import METHODS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="astro-stack", description="Align and stack astrophotography frames.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    st = sub.add_parser("stack", help="align and stack a folder of light frames")
    st.add_argument("lights", help="folder with the light frames (JPG, PNG, TIFF, FITS)")
    st.add_argument("-o", "--out", default="stack-output", help="output folder (default: stack-output)")
    st.add_argument("--darks", help="folder with dark frames (same exposure, lens covered)")
    st.add_argument("--flats", help="folder with flat frames (evenly lit, same focus)")
    st.add_argument("-m", "--method", choices=METHODS, default="sigma", help="combination method (default: sigma)")
    st.add_argument("--kappa", type=float, default=3.0, help="clipping threshold in sigmas (default: 3)")
    st.add_argument("--reference", help="file name of the reference frame (default: sharpest frame)")
    st.add_argument("--max-fwhm-ratio", type=float, default=1.5, help="reject frames blurrier than this times the median FWHM")
    st.add_argument("--min-stars", type=int, default=8, help="frames with fewer stars are not used to judge sharpness")
    st.add_argument("--no-crop", action="store_true", help="keep the edges that not every frame covers")

    dm = sub.add_parser("demo", help="generate a synthetic dataset and stack it")
    dm.add_argument("-o", "--out", default="astro-stack-demo", help="where to write data and results")
    dm.add_argument("--docs-dir", default="docs", help="where to write before/after images (default: docs)")
    dm.add_argument("--no-docs", action="store_true", help="do not write the README images")
    dm.add_argument("-n", "--frames", type=int, default=20, help="number of light frames (default: 20)")
    dm.add_argument("--seed", type=int, default=7)
    dm.add_argument("-m", "--method", choices=METHODS, default="sigma")

    args = parser.parse_args(argv)
    try:
        if args.command == "stack":
            from .pipeline import run

            run(
                args.lights,
                args.out,
                darks_dir=args.darks,
                flats_dir=args.flats,
                method=args.method,
                kappa=args.kappa,
                reference=args.reference,
                max_fwhm_ratio=args.max_fwhm_ratio,
                min_stars=args.min_stars,
                crop=not args.no_crop,
            )
        else:
            from .demo import run_demo

            if args.frames < 4:
                parser.error("the demo needs at least 4 frames")
            run_demo(args.out, None if args.no_docs else args.docs_dir, args.frames, args.seed, args.method)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
