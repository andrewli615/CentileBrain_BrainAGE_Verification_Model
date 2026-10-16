# Training data plot audit

Verified 317 PNG files: 237 display training data or depend on a training reference, and 80 contain no training data.

Passed 3930 numerical and provenance checks. Image integrity was checked for every PNG.

Checks cover sex/age selection, raw and adjusted reference values, BrainAGE_S formula, scatter coordinates, global and segmented regressions, Gaussian ridge curves, reference counts/means, bootstrap and permutation intervals, and source checksums.

- Training ridges use a Gaussian approximation; labels now explicitly say Gaussian fit.
- BrainAGE_S global training slopes are zero by construction; local segments need not be flat.
- Bucket-adjustment plots use RAW training; Chinese plots show raw or training-bucket-adjusted values. Duplicate diagnostic removed.
- Chinese corrections, coefficients, observed regressions, MAEs, and renderer coordinates were checked against source predictions.
- The repository contains exported predictions, not the fitted CentileBrain prediction model; upstream inference cannot be reproduced here.
- Bootstrap/permutation use original adjusted references, not BrainAGE_S.
- Q-Q plots use a theoretical normal reference, not training controls.
- Pixel images do not retain source coordinates; numerical verification uses saved summaries, source checksums, and current renderer artists.
