# Computational methods

The accompanying manuscript and Supplementary Methods give the full derivations. This guide connects the method to its implementation and saved evidence.

## Inputs and prediction

The release starts from the ordered abundance matrix (1,090 species by 2,894 samples) and fixed species annotations (463 coordinates). Presence gates each transformed annotation vector; abundance is a prediction target. Positive annotation values undergo log-one-plus transformation and training-occurrence-weighted centering/scaling. Zeros stay zero. `src/common.py` implements the shared model and transformation; `src/train_models.py` implements training and checkpoint selection. Configuration and partitions are saved with the results.

Six resource-internal predictors combine two representations and three training seeds. A seventh predictor holds out the entire verified AsnicarF_2021 study. The internal split has 34 known subjects crossing partitions; the study holdout excludes ambiguous metadata and separates study-scoped subjects. The separately archived same-seed replay is a reproducibility check, not an additional independent seed.

## Carrier deletion game

For a selected sample and coordinate, players are present species carrying that coordinate. A coalition replaces the selected standardized coordinate for its members, leaving their other coordinates unchanged. The game value is the distance between the original and perturbed predicted compositions. The main baseline is zero in standardized coordinates, which is distinct from deleting an entire species or its full pathway in a biological experiment. `src/real_analysis.py` constructs these games.

The target is the Shapley value of this defined game: each player's marginal response averaged over all permutations. Negative contributions are retained. Abundance and abundance-times-annotation allocations distribute an endpoint response without querying individual model effects. They have a different construction and are not assumed to estimate exact contributions accurately.

## Exact repeated-token computation

Absent species have identical input vectors in the tested architecture. Without a position signal, sequence-specific mask or stochastic dropout, their hidden states remain identical. `src/common.py` groups these identical tokens, adds log multiplicity to attention key logits and accounts for multiplicity in the final output softmax. Expanding the grouped probability recovers the dense model, including its absent-species mass. Grouping does not threshold away small probabilities. `src/verify_core.py` compares outputs, permutation equivariance and gradients.

## PCCV

`src/cv_shapley.py` exposes `cv_shapley(game, m, budget_permutations, seed, degree=3)`. `game` accepts a batch of binary coalition masks and returns one scalar per row. The implemented comparison domain is integer m >= 3 and nominal budget M >= 12 divisible by four. Degree two is a recorded ablation; degree three is PCCV.

The algorithm queries m singleton masks, m complements, the full coalition and eight fixed intermediate masks. Endpoint responses determine nonnegative normalized weights w. A least-squares fit with the implemented regularization constructs

    h(z) = a^T z + b (w^T z)^2 + c (w^T z)^3.

Its exact Shapley contribution for player i is

    phi_h[i] = a[i] + b*w[i]*W
               + c*(w[i]*W^2 + 0.5*w[i]*Q - 0.5*w[i]^2*W),
    W = sum(w), Q = sum(w^2).

The zero-weight fallback leaves the higher-order terms inactive. The implementation samples M-4 permutations in reverse pairs and evaluates the true game and surrogate on their prefixes. Two folds of permutation pairs estimate a clipped global control coefficient on the opposite fold. Each fold subtracts the centered surrogate contribution from the true marginal contribution; their combined mean estimates the same Shapley target as the sampler.

The true query-row count is (M-4)(m+1)+2m+9, bounded by M(m+1) in the supported domain. Repeated rows count as evaluations in this budget; calibration is included. Exact grouping accelerates each model query and benefits either estimator.

## Evaluation and evidence

- Ninety controlled games comprise fifty eight-carrier games and forty twelve-carrier games; nonnull mechanisms contain half planted null players.
- Forty fitted controlled predictors separately test the difference between prediction approximation and explanation accuracy.
- The frozen real-case plan defines eighteen samples and six coordinates, giving 108 cases. All seven allocation or attribution rules are retained.
- Six real conditional eight-carrier subgames have exact references. Unselected carriers stay unchanged, so these are not exact solutions to their full games.
- Independent sampling, baseline/distance changes, output permutation, endpoint agreement across training seeds, clustered bootstrap summaries and eighteen study-holdout explanation cases characterize sensitivity.
- `src/poly_baseline.py` is an independently implemented, explicitly configured polynomial baseline; the paper does not claim a complete benchmark of every setting in the official package. OddSHAP and ControlSHAP are methodological precedents, not separately executed head-to-head baselines.

`src/verify_reproduction.py` reloads seven real predictors and forty controlled predictors and reruns three complete real cases: fifty checks. Publication display verification independently recomputes plotted values and all 78 ranks in the illustrative case. `run_pipeline.py` isolates execution outputs from references.

These scores describe model responses to coordinate replacement. They do not establish measured pathway activity, species causality, disease mechanisms or fermentation performance. Full-game real PCCV superiority over permutation sampling is not established. Exact conditional and controlled references provide specified accuracy targets for reproducible comparison.
