"""Canonical MSU preprocessing entry point; see docs/MSU_PREPROCESSING.md.

The legacy CSV/stem-ID output is retired. Default mode validates annotations
without decoding; output directories must be new.
"""

from identity_invariant_fas.data.msu_preprocessing import main


if __name__ == "__main__":
    main()
