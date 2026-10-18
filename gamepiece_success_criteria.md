# Ideal world:
1. The projection function perfectly predicts the field coordinates that correspond to each pixel, enabling a constant heatmap (barring actual movement of the gamepieces)
2. If a cluster is identified, it remains identified as long as the centroid remains visible (i.e. no flickering clusters)
3. If an actually new cluster enters the image view, it should be identified as a new cluster rather than being forced into an existing cluster and raising the covariance dramatically
4. Clusters maintain semantic value and are not collapsed or introduced simply to minimize kmeans loss. They are meant to identify clusters of gamepieces and evaluate their location, density, and spread

# Issues:
1. We cannot individually identify gamepieces, so we use the proxy of pixels, but you cannot truly know the height of a pixel. This is why the existing algorithm is in place to better estimate the height, but it may be flawed.
2. This one should be relatively easy, just keep track of centroid location. However as soon as any other part of the algorithm falls apart, or a centroid is initialized in the wrong area it is difficult to keep a lineage of cluster IDs
3. It is difficult to separate a cluster that is naturally spread apart and which is just entering the frame vs a cluster which should actually be split because it's trying to cover too much ground and losing spatial information
4. There is a balance between going for too many and too few clusters; with too many clusters passed to kmeans, it will go gamepiece-by-gamepiece, which is useless for identifying clusters. If there are too few clusters, all spatial information is lost because gamepieces that are far apart will be grouped together

# Validation
1. Run exclusively on simulated data (using --fuel-clusters and then graduate to --fuel_noise when you can work in the ideal environment well enough)so that you know the ground truth. 
2. The ideal projection function results in the shape of the ball distribution being closely preserved.
3. The number, location, and shape of predicted clusters are close to ground truth

# Potential starting points (DON'T limit yourself to this):
1. Gamepieces outside of the neutral zone are not of concern, and anything projected there should be culled
2. Replace covariance spike with something more concretely geometric

# Execution
1. Make a worktree to do commits on, and create a frozen worktree copy every time you achieve significant progress. Do not commit on this branch
