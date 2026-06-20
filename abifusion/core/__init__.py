"""Portable, I/O-free fusion core.

This subpackage contains the language-neutral heart of abifusion: the
deterministic fusion logic (`fuse_abi`) plus the curated data tables it operates
on. Nothing here imports ``evmole``, ``requests``, ``torch``, or touches the
network — all of that lives in the adapters (see ``abifusion/ports.py``). Keeping
this boundary clean is what makes a future TypeScript/Rust port mechanical: a
port reimplements the three adapters and this one pure function, then validates
against the shared conformance vectors in ``tests/conformance/``.
"""

from abifusion.core.fuse import (
    choose_candidate,
    fuse_abi,
    parse_signature,
    split_args,
)

__all__ = ["fuse_abi", "choose_candidate", "parse_signature", "split_args"]
