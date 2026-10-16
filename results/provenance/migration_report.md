# Migration and validation report

Preserved 421 historical files byte-for-byte and verified all 210 protected source data files unchanged.

Finder metadata changes (not analysis inputs): ['ethnoracial_data_outputs/.DS_Store']. These were not restored or edited by the migration.

Generated five-method comparisons for 22 available panels. Passed 5926 numerical, source, archive, plot, and workbook checks.

## Locations

- Sources: original three directories, unchanged.
- Historical snapshot: archive/pre_reorganization/.
- Old-to-new map and hashes: archive/pre_reorganization/migration_inventory.json.
- Workbook and summary tables: results/summaries/.
- Participant values: results/participants/.
- Correction coefficients: results/coefficients/.
- Matched-population figures: results/plots/.
- Preserved supporting analyses: results/analyses/.

## Coverage and limitations

Six unavailable sex/age panels are recorded in coverage.csv. Six age-40 Chinese export rows are excluded from older panels. Unsupported corrections, exclusions, unmatched exports, and training support flags are retained in participant tables and coverage.csv. The upstream prediction model is external. Corrected MAEs are descriptive gaps, not independent prediction-validation errors. Historical files may document older paths; active manifests have been rebased. The source catalog is pinned and requires explicit review to change.
