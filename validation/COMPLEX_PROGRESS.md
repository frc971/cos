Complex validation in progress (maximum six piles)
=================================================

The earlier completion claim was not supported by sufficient cluster-count
coverage. The follow-up seven-pile diagnostic found real overmerging. Current
work follows the revised scope through six piles; seven-pile reports remain
historical evidence, not current acceptance runs.

First verified milestone: increasing the required density saddle from 55% to
75% of the smaller peak preserves modes separated by a 25% density valley.
This prevents distinct piles merging before persistent IDs are available;
cluster count remains inferred from density, with no fixed K or count limit.

At this milestone all 156 ideal four/five/six-pile frames and 78 ideal
six-pile mixed-density/elongated-pile frames pass. All 360 earlier regression
frames also pass. A six-pile cold-start unit regression passes, along with
the other ten tracker tests. Physical error gates remain 0.35 m centroid and
0.30 m shape, checked on every matched observation, with no count errors or
ID switches permitted.

The 132-frame wide camera sweep is still failing: 22 missing group
observations plus some geometry errors. Edge visibility and the inherited
minimum-image-pixel filter need investigation. Noise validation and the final
requirement audit are pending. The overall goal is not complete.

Evidence: `evidence/complex-baseline.json`, `complex-ideal.json`,
`complex-mixed-ideal.json`, `complex-dynamic-ideal.json`, and the three
`prominence-regression-*.json` files. Fixture files start with `complex_`.
Target artifacts are under `/root/cluster-validation/` in
`complex-baseline`, `complex-prominence`, `complex-mixed-ideal`,
`complex-dynamic-ideal`, and `prominence-regression`.
