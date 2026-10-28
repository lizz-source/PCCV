**Figure1_exact_sparse_inference.** Exact repeated-token inference. Eighteen richness-stratified communities; error is against original dense attention; CPU times are medians of five repeats after warmup.

**Figure2_ground_truth_attribution.** Attribution against exact coalition values in controlled response mechanisms. Panel A uses a symmetric log scale with a linear region within 0.05. Null carriers have no effect on the response generator. These are explanation benchmarks, not biological communities.

**Figure3_matched_budget_estimation.** Matched-query comparison of antithetic Shapley sampling and PCCV. Ten independent mechanism seeds and ten sampling replicates per family/budget. The compensatory family is retained as a failure boundary.

**Figure4_predictor_controls.** Freshly trained composition predictors. Panel A uses a resource-internal sample split and contains known repeated-subject overlap; panel B holds out an entire study with subject-disjoint training/validation. Error bars in A show training-seed SD.

**Figure5_real_perturbation_faithfulness.** Deletion and restoration curves in all preselected test cases, averaged after normalization by each case joint response. Curves may exceed one under nonmonotone effects. These are interventions on model inputs.

**Figure6_exactness_and_sampling.** Six exact eight-carrier conditional subgames and independent sampling agreement across full carrier sets. Other carriers are unchanged in exact subgames.

**Figure7_carrier_context_profiles.** Signed PCCV contributions for the fourteen largest mean-absolute carriers in the selected highest-response pathway. These profiles describe model dependence and do not establish ecological keystoneness.

**Figure8_larger_games_and_kernel_baseline.** Twelve-carrier games with all 4096 coalitions providing exact references, using ten new mechanism seeds and ten sampling replicates. A log error scale displays all five estimators. KernelSHAP uses kernel-proportional coalition sampling. PolySHAP-3 is an independent Algorithm 1 implementation with all 298 terms and paired/unpaired kernel-proportional sampling; it is evaluated at 32/64 budgets, where the budget exceeds the number of terms. No leverage or adaptive-frontier implementation is evaluated.