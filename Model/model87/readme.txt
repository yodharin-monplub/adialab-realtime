Model87: exact incremental deployment repair for the timed-out Model73 bundle.

Cloud run 116613 failed after 9h52m52s with exit code 124. The deployed
SeriesPredictor called whitening_features(history, online_prefix) at every
online observation. That repeatedly refit history-only preprocessing and
recomputed every prior prefix, producing quadratic runtime in series length.

Model87 adds StreamingWhitening. It fits the history-only AR model, historical
EWMA variance path, empirical normal reference, channel normalization, and
context once per series. Each step then updates the AR residual, EWMA state,
empirical rank, and cumulative window sums in constant time. The frozen batch
implementation remains the independent parity oracle. No model, feature
definition, coefficient, or prediction recipe is intentionally changed.

Validation completed successfully. Across 3,248 whitening rows and 540
end-to-end prediction rows, the incremental and failed-bundle implementations
are bitwise identical (maximum error 0), including under cloud Python 3.12.
Incremental generator consumption and inference-only training checks passed.

On the same 24 series / 13,528 rows / 2 workers, Model87 took 141.98 wall
seconds and 269.14 summed worker seconds versus the successful Model26
baseline's 156.68 and 299.71 seconds. Anchoring to Model26 cloud run 114949
(20,237 seconds) gives a 5.62-hour nominal estimate and a conservative
4.5--7.3-hour planning range. Twelve workers project to about 5.46 GiB of
model-worker RSS, excluding parent/platform overhead, within the requested
16-CPU/64-GiB setup.

The sealed, hash-verified 83,412,345-byte archive is submission28.zip, SHA-256
310102512c816d8b51cb539a13ecd51fbaad0ac5bbf3c4a63bacff26e8c3f925.
It has not been uploaded and no new cloud run has been started.
