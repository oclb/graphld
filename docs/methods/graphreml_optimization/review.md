# Integration review

The review covered covariance reconstruction and rejected-step restoration, current-factor precision solves, node-level trace corrections, global information-penalty values and gradients, optimizer proposal safeguards and actual-objective line searches, precise endpoint checks, and scaled unregularized jackknife solves. Synthetic validation includes 193 passing checks and 29 overlapping metadata checks. The methodological write-up reports independent validation of the saved real-data endpoints and aggregate enrichment calculations.

## Open default decision

The tested BFGS default is not supported for release by these comparisons. Weight remained nonstationary after 100 iterations, despite numerical safeguards; AI converged from the same start. Both methods converged on BMI, where BFGS took longer. The branch retains the requested BFGS default while this decision is reviewed. This is a scientific acceptance issue even though the implementation tests pass.

## Penalty and uncertainty

The optional information penalty remains off by default. AI penalty-off/on fits passed the endpoint checks for Weight and Hemoglobin. Exact trial evaluation remains the supported default after a matched screening comparison showed no runtime benefit. Finite-difference and synthetic checks support the analytic penalty gradient; these results do not establish formal bias reduction for this approximation.

The repaired pseudo-jackknife uses likelihood information at the returned endpoint. Penalized uncertainty calibration and the effect of omitting penalty curvature are deferred as specified for this change. Near-equal objective values can coexist with larger relative differences in a depleted annotation, as shown in the working-control and screening results.

## Final endpoint and source checks

The historical BMI endpoint was reproduced exactly and failed the precise criteria (Q=0.8057; tested likelihood gain=0.7132), with zero parameter updates. All 22 production source files matched the frozen numerical runtime. The final changes after numerical evaluation were documentation corrections and links. The two saved test reports were independently hash-checked and parsed; their overlapping counts are kept separate in [the validation manifest](validation_manifest.json). The packaged fourteen-page LaTeX document compiled cleanly, and the rendered figures and affected pages were visually inspected.
