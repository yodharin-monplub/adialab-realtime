Model160 - more oracle channels for the teacher (2026-09-22).

oracle_v2.py: 6 extra privileged features - the Bayesian break posterior on
the RAW-level normal scores (history ECDF, no AR whitening), its scale-only
version, a lag-2 dependence posterior on the whitened stream, and a 4-channel
mixture. Single columns: raw_scale_cdf .634, dep2_cdf .600, mix4_cdf .673.
Teacher probe (fold 0, j=2): 0.7190 -> 0.7291; teacher D (old 25 + oracle 7 +
these 6, run_model157.py TVAR=D) fold 0: .729/.704/.727 vs teacherB
.719/.699/.724.
Student SRD40 vs SRB40 (dev): -0.0049 +0.0027 -0.0004 +0.0011 -0.0003
  (mean -0.0004).
Sealed holdout: student 0.62884 vs SRB40 0.62950, final 0.63430 vs 0.63449.
-> a teacher with higher AUC is not automatically a better target-maker
   (same as the bigger teacher C in Model157). CLOSED; teacherB stays.
