# Changelog

All notable changes to this project will be documented in this file.

## [v0.2.1] - 2024-08-13

### Added

- Added the `--max_obs_trace` argument to the parser. This can be used to manually set the maximum number of observations to trace. Useful for rerunning with a larger value on a subset of (previouslytruncated) reads. This overrides the value in the config file.
- When the `min_obs_adapter` value is detected as the `adapter_end` in the llr workflow, a second round of llr gains (starting from `min_obs_adapter`) is calculated to refine the `adapter_end` detection.
- Detect results contain information on full read length and preload size.
- Added a `debug_logger` object in the llr workflow.
- Added a `refine_max_adapter_end_adjust` parameter to the llr workflow configuration. When the suggested adjusted in the refinement is more than this value, the refinement is skipped. Default is 250.
- Added a `--config` flag to the parser. This can be used to specify the latest default config TOML file.

### Fixed

- Signal preload: signal preload size is now explicitly updated based on the `max_obs_trace` value after initializing the config object.
- Partition stats: partition (adapter, polya, preloaded_rna) stats are now computed for the adapter based on the llr-detected boundary, and no longer becomes `None` when the mvs workflow fails in the combined detection.
- setup.py: fixed the package import to correctly find all packages and sub-packages.
- setup.py: added a `config_files` folder to the package data.

### Changed

- Renamed `rna` partition stats (start, len, mean, std, med, mad) to `preloaded_rna` to clarify that they are computed from the preloaded RNA signal, not the full read signal.
- `adapter_end` defaults to the value detected in the llr workflow, also when the mvs workflow fails in the combined detection (previously defaulted to `None`).
- `preloaded_rna` partition stats are `None` in the case of truncated polya signals.
- Median shift calculation in mvs workflow is now done using the median of the detected adapter signal as a whole, instead of the median of a specified window.
- The ADAPTed version is now part of the created output dirctory name.
- The parser now requires a valid configuration file (`--config`) or a chemistry (`--chemistry`) to be provided.
- Parameters for RNA002:
  - `min_obs_adapter` = 2000 (previously 2500)
  - `max_obs_trace` = 12000 (previously 8000)
  - `adapter_trace_stride` = 50 (previously 40)
  - `adapter_trace_early_stop_stride` = 500 (previously 1000)
  - `polya_trace_early_stop_stride` = 50 (previously 40)
  - `polyA_local_range` = [0.0, 12.0] (previously [0.0, 10.0])
