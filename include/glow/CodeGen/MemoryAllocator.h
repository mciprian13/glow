/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef GLOW_CODEGEN_MEMORYALLOCATOR_H
#define GLOW_CODEGEN_MEMORYALLOCATOR_H
#include <cstddef>
#include <cstdint>
#include <list>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "glow/Support/Memory.h"

namespace glow {

/// Utility function to verify if two contiguous integer intervals overlap. Each
/// interval is described by a half-open interval [begin, end).
template <class T>
inline bool intervalsOverlap(T begin1, T end1, T begin2, T end2) {
  return (std::max(begin1, begin2) < std::min(end1, end2));
}

/// Type that should be used as a handle for memory segments for identification.
/// TODO: Replace the Handle from allocator with this one.
using MemoryHandle = const void *;

/// Allocation structure which represents a request to allocate or free
/// a memory region (buffer) within a given MemoryAllocator instance.
struct Allocation {

  /// Allocation handle which uniquely identifies the buffer to be allocated.
  /// The provided handle must be unique for each buffer.
  MemoryHandle handle_;

  /// Allocation request type flag: true for ALLOC, false for FREE.
  bool alloc_;

  /// Allocation buffer size in bytes. This field is mandatory for ALLOC
  /// and is ignored for FREE.
  uint64_t size_;

  Allocation(MemoryHandle handle, bool alloc, uint64_t size)
      : handle_(handle), alloc_(alloc), size_(size) {}

  Allocation(size_t id, bool alloc, uint64_t size)
      : handle_((MemoryHandle)(id)), alloc_(alloc), size_(size) {}
};

/// A POD struct that represents a single half-open allocation [start .. end).
struct Segment {

  /// The allocation starts at this address.
  uint64_t begin_;

  /// The allocation ends before this address (half-open interval).
  uint64_t end_;

  Segment(uint64_t begin, uint64_t end) : begin_(begin), end_(end) {}

  /// \returns the size of the interval.
  uint64_t size() const { return end_ - begin_; }

  /// \returns True if the value \p idx falls within this segment.
  bool contains(uint64_t idx) const { return idx >= begin_ && idx < end_; }
};

/// Allocates segments of memory.
/// Each allocation is associated with a user-defined handle, typically
/// representing a client-specific object, e.g. a handle can be a `Value *` and
/// represent a value whose payload is going to be stored in the allocated
/// memory block. This simplifies the clients of MemoryAllocator and allows them
/// to use higher-level client-side objects instead of raw allocated addresses
/// to refer to the allocated memory blocks.
class MemoryAllocator {
public:
  /// Type that should be used as a handle.
  using Handle = const void *;

  /// A reserved value to mark invalid allocation.
  static const uint64_t npos;

  explicit MemoryAllocator(const std::string &name, uint64_t memorySize,
                           size_t alignment = TensorAlignment)
      : name_(name), memorySize_(memorySize), alignment_{alignment} {}

  void reset() {
    liveSize_ = 0;
    maxUsedSize_ = 0;
    maxLiveSize_ = 0;
    segments_.clear();
    handleToSegmentMap_.clear();
    addrToHandleMap_.clear();
  }

  /// \returns True if the value \p idx is within the currently allocated range.
  bool contains(uint64_t idx) const {
    for (auto &s : segments_) {
      if (s.contains(idx)) {
        return true;
      }
    }
    return false;
  }

  /// Allocate a segment of size \p size and associate a \p handle with it.
  /// \returns the allocated pointer, or MemoryAllocator::npos, if the
  /// allocation failed.
  uint64_t allocate(uint64_t size, Handle handle);

  /// Allocate a segment of size \p size and associate a handle \p Handle with
  /// it. If the allocation is not possible, the allocator should try to evict
  /// some entries that are not needed at the moment, but it is not allowed to
  /// evict any entries from \p mustNotEvict set. All evicted entries are stored
  /// in the \p evicted set.
  /// \returns the allocated pointer, or MemoryAllocator::npos, if the
  /// allocation failed.
  uint64_t allocate(uint64_t size, Handle handle,
                    const std::set<Handle> &mustNotEvict,
                    std::vector<Handle> &evicted);

  /// Allocate all the segments associated with the allocations \p allocArray.
  /// This method has an improved memory allocation efficiency because all
  /// the allocations are requested at once and the algorithm can improve the
  /// allocation efficiency by allocating first the larger segments and so
  /// avoiding early fragmentation. Upon function return use ONLY the function
  /// \ref getSegment to retrieve the allocated segments based on the handles
  /// used in \p allocArray.
  /// \returns the total memory usage, or MemoryAllocator::npos, if the
  /// allocation failed.
  uint64_t allocate(const std::vector<Allocation> &allocArray);

  /// \returns the handle currently associated with the allocation at \p
  /// address.
  Handle getHandle(uint64_t ptr) const;

  /// \returns true if there is a handle currently associated with the
  /// allocation at \p address.
  bool hasHandle(uint64_t ptr) const;

  /// \returns the segment currently associated with the \p handle.
  Segment getSegment(Handle handle) const;

  /// \returns the address currently associated with the \p handle.
  uint64_t getAddress(Handle handle) const;

  /// \returns the size of the allocated block currently associated with the \p
  /// handle.
  uint64_t getSize(Handle handle) const;

  /// \returns true if there is an address currently associated with the \p
  /// handle.
  bool hasAddress(Handle handle) const;

  /// Frees the allocation associated with \p handle.
  void deallocate(Handle handle);

  /// \returns the total size (in bytes) of the memory.
  uint64_t getMemorySize() const { return memorySize_; }

  /// \returns the maximum memory usage (in bytes).
  uint64_t getMaxMemoryUsage() const { return maxUsedSize_; }

  /// \returns the alignment boundary used to align segments.
  size_t getAlignment() const { return alignment_; }

  /// \returns the allocation efficiency as a float between 0.0 and 1.0
  /// which quantifies the efficiency of the allocation algorithm. An
  /// efficiency value of 1.0 means the best theoretically possible. The
  /// efficiency is not always 1.0 due to memory fragmentation. 
  float getAllocationEfficiency() const;

  /// \returns the name of the memory region.
  const std::string &getName() const { return name_; }

private:
  /// The name of the memory region.
  std::string name_;

  /// A list of allocated segments.
  std::list<Segment> segments_;

  /// The total size (in bytes) of the memory region. A value of 0 means unlimited size (infinite).
  uint64_t memorySize_;

  /// The maximum size (in bytes) used for segment allocation (memory usage).
  uint64_t maxUsedSize_{0};

  /// The maximum size (in bytes) for all simultaneously alive segments during
  /// allocation. This is a theoretical size and is the best (minimum) memory
  /// usage we can get with any allocation algorithm since it ignores memory
  /// fragmentation. Since maxLiveSize_ <= maxUsedSize_ we can use this to
  /// compute an allocation efficiency as maxLiveSize_ / maxUsedSize_.
  uint64_t maxLiveSize_{0};

  /// Current size (in bytes) for all the live segments currently allocated.
  /// This is automatically updated for each call to \ref allocate or
  /// \ref deallocate.
  uint64_t liveSize_{0};

  /// The alignment boundary for each segment allocation.
  size_t alignment_;

  /// Maps allocated addresses to the currently associated handles.
  std::unordered_map<uint64_t, Handle> addrToHandleMap_;

  /// Maps handles to the allocation information about the memory block
  /// currently associated with them.
  std::unordered_map<Handle, Segment> handleToSegmentMap_;

  /// \returns the effective size used for allocating a segment which depends
  /// on the requested allocation size \p size and the alignment used by this
  /// allocator instance.
  uint64_t getEffectiveSize(uint64_t size) const;

  /// Tries to evict some entries that are not needed at the moment to free
  /// enough memory for the allocation of \p size bytes, but it is not allowed
  /// to evict any entries from \p mustNotEvict set. All evicted entries are
  /// stored in the \p evicted set. Uses first-fit approach for finding eviction
  /// candidates.
  void evictFirstFit(uint64_t size, const std::set<Handle> &mustNotEvict,
                     std::vector<Handle> &evicted);

  /// Associates a \p handle with an allocated address \p ptr and size \p size.
  void setHandle(uint64_t ptr, uint64_t size, Handle handle);

  /// Function to verify the allocation requests before allocating the segments.
  /// \returns true if allocations are valid and false otherwise.
  bool verifyAllocations(const std::vector<Allocation> &allocArray);

  /// Function to verify the segments after allocating them.
  /// \returns true if segments are valid and false otherwise.
  bool verifySegments(const std::vector<Allocation> &allocArray);
};

} // namespace glow

#endif // GLOW_CODEGEN_MEMORYALLOCATOR_H
