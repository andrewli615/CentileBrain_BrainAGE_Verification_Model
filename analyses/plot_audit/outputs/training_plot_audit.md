# Training data plot audit

Verified 318 PNG files: 238 display training data or depend on a training reference, and 80 contain no training data.

Passed 3700 numerical and provenance checks. Image integrity was checked for every PNG.

Checks cover sex/age selection, raw and adjusted reference values, BrainAGE_S formula, scatter coordinates, global and segmented regressions, Gaussian ridge curves, reference counts/means, bootstrap and permutation intervals, and source checksums.

- Training ridges use a Gaussian approximation; labels now explicitly say Gaussian fit.
- BrainAGE_S global training slopes are zero by construction; local segments need not be flat.
- Bucket-adjustment main plot uses RAW training with ADJUSTED Chinese; diagnostic uses both adjusted.
- Bootstrap/permutation use original adjusted references, not BrainAGE_S.
- Q-Q plots use a theoretical normal reference, not training controls.
- Pixel images do not retain source coordinates; numerical verification uses saved summaries, source checksums, and current renderer artists.
