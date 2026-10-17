# e55 M5 Pro synthetic evidence

Final window: `vision-20261017T142545Z`; earlier failures and pre-fix measurements remain separate.
Full-model measurements use the separate local resident integration. The upstream-only tree has independent tiny-model tests; resident serving and PR11 are not part of the upstream vision patch.

`REPORT.md` gives implementation, test profiles, failures and limits. `source-manifest.json` pins every tested core module. The source tree in this commit matches its corresponding manifest entry. Original source archives and wiki overlays remain in the parent MacMini project evidence archive; embedding repositories inside their own result tree is unnecessary.

`executed-scripts` holds exact machine-specific clients/guards. The parameterized replay is `bench/experiments/e55_vision_integration.py`; give it the pinned official root and a separately prepared config while holding exclusive workload locks. The HTTP client does not pause photos or restore production on its own. Synthetic fixture pixels are included. No private image inputs or ordinary serving outputs.
