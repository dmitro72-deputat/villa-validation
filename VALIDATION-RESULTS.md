# Villa validation results

## Anisotropic patch discovery
The proposed patch computes independent Y and X strides in labeled and unlabeled patch discovery.

Real CT experiment: PHercParis4, uint8 crop [1024:1040,1408:1664,1408:1664], shape (16,256,256).
SHA-256: c3167593f37f1ce2f87dba3048237e676462a0e6b4860b98f74586a271fe4840.

| Y:X scale | Original patches | Patched patches | Original covered support | Patched covered support |
|---|---:|---:|---:|---:|
| 1:1 | 148 | 148 | 37825/37825 | 37825/37825 |
| 1:4 | 45 | 149 | 20734/37825 | 37825/37825 |
| 4:1 | 564 | 149 | 37825/37825 | 37825/37825 |

At 1:4, coverage increases from 54.82% to 100%. At 4:1, patch count decreases by 73.58% without reducing measured coverage.

[Reproducible CI experiment](https://github.com/dmitro72-deputat/villa-validation/actions/runs/45819162053)

Limitations: artificial anisotropic subsampling of one real CT crop, nonzero intensity support rather than annotated papyrus or ink, lightweight dependency stubs, unlabeled discovery only in this experiment. No claim about transcription accuracy or full-scroll performance.

## Cache optimization
A separate patch batches JSON serialization and uses atomic replacement. Previously reported synthetic benchmark figures require independent reproduction before being treated as final submission evidence.

## Regression tests
Both patches applied together: 126 passing existing tests.
[CI run](https://github.com/dmitro72-deputat/villa-validation/actions/runs/45813972391)
Tested fork commit: 6b9dd4d1c8d85fd8851bda1762d767f300131da1.

## Review status
Candidate contributions, not yet accepted upstream. Exact upstream base SHA and award eligibility remain to be confirmed. No prize or payment is claimed.
