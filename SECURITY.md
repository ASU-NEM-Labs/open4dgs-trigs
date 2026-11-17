# Security policy

## Reporting a vulnerability

Please report suspected vulnerabilities privately through
[GitHub's private vulnerability reporting form](https://github.com/ASU-NEM-Labs/open4dgs-trigs/security/advisories/new)
(Security → Advisories → Report a vulnerability). Do not post exploit details,
credentials, or private capture data in a public issue. If the private report
option is unavailable, open an issue requesting a private reporting channel
without including vulnerability details.

Include the affected repository and commit/tag, environment, impact, and minimal
reproduction steps. Remove private information from logs and samples. Reports
are reviewed on a best-effort basis; no response-time or fix-time guarantee is
provided. Ordinary installation problems and reconstruction-quality bugs belong
in the public issue tracker after sensitive information is removed.

## Supported code

Security fixes are intended for the current `main` branch. This initial research
release does not offer maintained older release branches or long-term support.

## Intended use and trust boundaries

These are local research tools, not a hardened service for processing arbitrary
untrusted uploads. Some checkpoint readers use `torch.load(...,
weights_only=False)`, and some NPZ readers allow pickled objects. Loading a
malicious file through those paths can execute code. Only load checkpoints,
model weights, and initialization files from sources you trust.

Run capture processing and model code with ordinary user privileges. Keep viewer
and development endpoints on a trusted local network or behind access controls;
do not expose them directly as public upload or processing services. Review
third-party model code and setup commands before running them.

Dataset permission and privacy remain the responsibility of the person providing
the data. Do not attach private footage, credentials, or identifying capture
metadata to public bug reports.
