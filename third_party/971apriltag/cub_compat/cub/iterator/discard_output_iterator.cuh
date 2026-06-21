// Compatibility stub: cub::DiscardOutputIterator was removed in CUDA 13.x CCCL.
// Redirect to the thrust equivalent.
#pragma once
#include <thrust/iterator/discard_iterator.h>

namespace cub {

template <typename OffsetT = ptrdiff_t>
using DiscardOutputIterator = thrust::discard_iterator<OffsetT>;

} // namespace cub
