# Frozen upstream provenance

This directory contains material inherited from the upstream SAMURAI fork at
commit `76ba195984892b0d1e3db5d9c9f90bb62175680a`. The SAM 2 subtree is a
Meta-derived snapshot; the `samurai/` subtree contains the fork's legacy
tracking/evaluation code.

It is retained as dated observation for comparison only. Canonical code under
`src/samurai/` uses SAM 3.1 and must not import either snapshot. The inherited
GitHub Actions workflow and tracked `.DS_Store` were removed; no source here is
patched for runtime behavior.
