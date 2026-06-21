// Compatibility stub: cub::TransformInputIterator was removed in CUDA 13.x CCCL.
// Redirect to the thrust equivalent.
#pragma once
#include <thrust/iterator/transform_iterator.h>

namespace cub {

template <typename ValueType, typename ConversionOp, typename InputIteratorT, typename OffsetT = ptrdiff_t>
using TransformInputIterator = thrust::transform_iterator<ConversionOp, InputIteratorT, thrust::use_default, ValueType>;

} // namespace cub
