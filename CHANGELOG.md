# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

### Added

- A new `fallback_to_llr` parameter to the `cnn_boundaries` and `rna_start_peak` sections of the config file. When set to true, the primary detection method will fallback to the LLR detection method upon failure.

### Removed

- Unused detection result attributes `polya_truncated`, `llr_adapter_end_adjust`, `llr_polya_end_adjust`, `llr_trace_early_stop_pos`, `mvs_llr_polya_end_adjust_ignored` and `mvs_llr_polya_end_to_early_stop`, and resulting columns in the output csv files, were removed.

## [v0.2.4] - 2024-12-09

### Changed

- Major redesign of the `file_proc` module. Reads are processed in minibatches. The `adapted.file_proc.file_proc` submodule has been removed and the functionality has been moved to `adapted.file_proc`.

### Fixed

- Non runnable detection workflow caused by partial code updates.

### Added

- A new `detect_rna_start_peak` and `combined_detect_start_peak` function that use the start peak detection method to provide a quick alternative for e.g. tRNA and custom adapter workflows.
- A new `med_shift` validation workflow that uses the median shift method between adapter and pos-adapter signal to validate the detected boundaries. Does not require a polyA tail signal.

## [v0.2.3] - 2024-11-13

### Fixed

- Mysterious runtime warnings occuring in the mvs polya workflow were identified as being due to too short signals. These signals are now filtered out before the detection workflow.
- Run time warnings during peak detection are now filtered.

### Added

- A new detection workflow is added that uses a CNN to predict the boundaries of the adapter and polyA signals. The CNN method replaces the previous llr workflow as the primary detection method and provides faster detection. The CNN workflow comes with a depency on torch (`combined_detect_cnn`).
- SigProcConfig now has a primary_config attribute that is set at runtime to describe the detection workflow used prior to validation.
- Introduction of a new `combined_detect_llr2` function that uses a downscaled signal for the llr workflow, with a split peak correction for the polyA trace.
- The `combined_detect_cnn` function now has a fallback to a quick version of the `llr` method if validation of the boundary predictions fails.

### Changed

- `config.batch.bidx_passed/failed` is now `config.batch.batch_idx_pass/fail`.
- default minibatch size is now 1000.
- `combined_detect` has been renamed to `combined_detect_llr`.
- `ReadResult`, `DetectResults` and `Boundaries` have been moved to `adapted.container_types`.
- Preloaded signals that are shorter than the preload size are now padded with nans instead of zeros.
- Mad winsorization by imputing outliers with local medians has been changed to mad winsorization by clipping.
- The flag for setting the `num_proc` in the parser has been changed to `-j`.
- Introduction of the 'core' SigProcConfig section that contains parameters that are used across multiple detection methods in the code.
- `sig_norm_outlier_thresh` has been moved to the new `core` section, as have adapter, trace and polya min/max obs parameters.
- The `llr_helpers` module has been merged into the `llr` module.

### Removed

- The Task attribute of the Config object is removed.
- The `save_llr_trace` parser argument is removed.
- Functions for mad winsorization by imputing outliers with local medians (`impute_window_median`, `mad_outlier_indices`, `mad_normalize`, `mad_winsor`) are removed.
- The `sig_norm_winsor_window` `LLRBoundariesConfig` attribute is removed.


## [v0.2.2] - 2024-09-10

### Fixed

- Faulty pod5 files (e.g. `pod5: IOError: Invalid signature in file`) are now catched, faulty input files are skipped.

### Added

- get_truncated.sh: added a new script to easily obtain the truncated reads from the detected_boundaries*.csv files.
- command.json file is now saved in the output folder and contains the command used to run ADAPTed.
- Logging: process outputs are now logged to the `adapted.log` file and to stdout.
- ADAPTed now supports continuing from a previous (incomplete) run using the `continue <continue_from_path>` subcommand.

### Changed

- Multiprocessing of reads now relies on a shared memory object to process the results of the reads. This allows for a more efficient use of system resources and prevents issues with broken pipe errors.
- The output directory is now named after the version of ADAPTed and a random UUID rather than the current date and time.

### Removed

- The `--create_subdir` argument is removed. The output directory is now always created in the specified output folder.

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
