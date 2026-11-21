#pragma once

#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace utils {

class DisjointSetUnion {
 public:
  using element_type = std::size_t;

  explicit DisjointSetUnion(std::size_t element_count)
      : parent_(element_count),
        component_size_(element_count, 1),
        component_count_(element_count) {
    std::iota(parent_.begin(), parent_.end(), 0);
  }

  DisjointSetUnion() : DisjointSetUnion(0) {}

  // Removes all elements while retaining the allocated storage for reuse.
  auto Clear() -> void {
    parent_.clear();
    component_size_.clear();
    component_count_ = 0;
  }

  // Adds a new singleton component and returns its element index.
  auto FillSets(std::size_t num_elements) -> std::vector<element_type> {
    for (size_t i = 0; i < num_elements; i++) {
      const element_type element = parent_.size();
      parent_.push_back(element);
      component_size_.push_back(1);
      ++component_count_;
    }
    return parent_;
  }

  auto Find(element_type element) -> element_type {
    CheckElement(element);

    element_type root = element;
    while (parent_[root] != root) {
      root = parent_[root];
    }

    while (parent_[element] != element) {
      const element_type next = parent_[element];
      parent_[element] = root;
      element = next;
    }
    return root;
  }

  [[nodiscard]] auto Find(element_type element) const -> element_type {
    CheckElement(element);
    while (parent_[element] != element) {
      element = parent_[element];
    }
    return element;
  }

  auto Union(element_type first, element_type second) -> bool {
    element_type first_root = Find(first);
    element_type second_root = Find(second);

    if (first_root == second_root) {
      return false;
    }

    // Keep the larger tree as the root.
    if (component_size_[first_root] < component_size_[second_root]) {
      std::swap(first_root, second_root);
    }
    parent_[second_root] = first_root;
    component_size_[first_root] += component_size_[second_root];
    --component_count_;
    return true;
  }

  // Returns whether first and second belong to the same component.
  auto Connected(element_type first, element_type second) -> bool {
    return Find(first) == Find(second);
  }

  [[nodiscard]] auto Connected(element_type first, element_type second) const
      -> bool {
    return Find(first) == Find(second);
  }

  // Returns the number of elements in the component containing element.
  auto ComponentSize(element_type element) -> std::size_t {
    return component_size_[Find(element)];
  }

  [[nodiscard]] auto ComponentSize(element_type element) const -> std::size_t {
    return component_size_[Find(element)];
  }

  // Returns the number of elements and components, respectively.
  [[nodiscard]] auto Size() const -> std::size_t { return parent_.size(); }
  [[nodiscard]] auto ComponentCount() const -> std::size_t {
    return component_count_;
  }

 private:
  void CheckElement(element_type element) const {
    if (element >= parent_.size()) {
      throw std::out_of_range("DisjointSetUnion element index out of range");
    }
  }

  std::vector<element_type> parent_;
  std::vector<std::size_t> component_size_;
  std::size_t component_count_;
};

}  // namespace utils
