# Reconstructed historical dependency

The frozen submission omitted `detector/channels18.py`, even though its unused `detectors19.py` and `detectors20.py` ablations import it. The copy included here comes from the current audited `detector/core/channels18.py`; it is clearly supplemental and is not claimed to be the lost historical revision. The deployed Model 1–3 feature contracts never imported those two ablations.
