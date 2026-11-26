# Upstream submission drafts

## PR 1 — Fix independent XY strides in flat patch discovery

### Problem
The default labeled and unlabeled patch finders use a shared stride derived from the Y dimension. This can miss support or generate redundant patches when scan scaling or patch dimensions differ between axes.

### Changes
Compute independent Y and X strides and use them for grid occupancy and patch origins. Add six regression cases covering anisotropic scaling and rectangular patches in both discovery modes.

### Validation
132 tests passed: https://github.com/dmitro72-deputat/villa-validation/actions/runs/45821326944

On one PHercParis4 CT crop with artificial anisotropic subsampling, support coverage at Y:X 1:4 increased from 54.82% to 100%; at 4:1 patch count decreased from 564 to 149 with unchanged measured coverage.
https://github.com/dmitro72-deputat/villa-validation/actions/runs/45819162053

This measures nonzero intensity support, not ink recognition or transcription accuracy.

Branch: fix/anisotropic-patch-stride
Tested commit: 3d22fc3dc8ef6b9f9150690263672ad8c5805db9

## PR 2 — Reduce patch-cache serialization allocations

### Problem
The writer materializes all JSON records before serialization.

### Changes
Serialize batches of 1024 records, reuse configuration tokens with a bounded cache, and replace the destination atomically.

### Validation
126 existing tests passed: https://github.com/dmitro72-deputat/villa-validation/actions/runs/45833689725

100,000 synthetic records, ten configurations, three repetitions, CPython 3.12.16:
- Median write time: 1.509116327 s to 0.354257077 s (4.26x).
- Median peak traced Python allocations: 62.605872154 MiB to 1.355607986 MiB (97.83% reduction).
- All measured files: 40067918 bytes, identical SHA-256 9cd4c9834deef8d044a9c0eee558dc5018ba31ec99ee20569e597e105e5ea7b1.

https://github.com/dmitro72-deputat/villa-validation/actions/runs/45836348251

Synthetic lightweight inputs; input allocation excluded; not process RSS or full-pipeline performance. Atomic replacement changes file identity and newly created files use temporary-file permissions; reviewers should assess compatibility.

Branch: perf/patch-cache-memory
Tested commit: aa60826e422dfe32ec8b01de1c8ab1c92f0247f9

## Submission status
Drafts only. Upstream acceptance and prize eligibility have not been established. AI-assisted development and validation should be disclosed in accordance with applicable contribution rules.
