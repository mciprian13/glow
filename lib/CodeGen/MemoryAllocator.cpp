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

#include "glow/CodeGen/MemoryAllocator.h"
#include "glow/Support/Debug.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "memory-allocator"

using namespace glow;

namespace glow {
class Value;
}

/// The type of the address returned by MemoryAllocator::allocate should be at
/// least 64-bit wide.
static_assert(sizeof(decltype(MemoryAllocator::npos)) >= 8,
              "Allocated addresses should be at least 64-bit wide");

/// The type of the address returned by MemoryAllocator::allocate should be
/// unsigned
static_assert(std::is_unsigned<decltype(MemoryAllocator::npos)>{},
              "Allocated addresses should be unsigned integers");

const uint64_t MemoryAllocator::npos = -1;

uint64_t MemoryAllocator::allocate(uint64_t size, Handle handle) {
  // Always allocate buffers properly aligned to hold values of any type.
  uint64_t segmentSize = alignedSize(size, alignment_);
  uint64_t prev = 0;
  for (auto it = segments_.begin(), e = segments_.end(); it != e; it++) {
    if (it->begin_ - prev >= segmentSize) {
      segments_.emplace(it, prev, prev + segmentSize);
      maxMemoryAllocated_ = std::max(maxMemoryAllocated_, prev + segmentSize);
      setHandle(prev, size, handle);
      return prev;
    }
    prev = it->end_;
  }
  // Could not find a place for the new buffer in the middle of the list. Push
  // the new allocation to the end of the stack.

  // Check that we are not allocating memory beyond the pool size.
  if (poolSize_ && (prev + segmentSize) > poolSize_) {
    return npos;
  }

  segments_.emplace_back(prev, prev + segmentSize);
  maxMemoryAllocated_ = std::max(maxMemoryAllocated_, prev + segmentSize);
  setHandle(prev, size, handle);
  return prev;
}

void MemoryAllocator::evictFirstFit(uint64_t size,
                                    const std::set<Handle> &mustNotEvict,
                                    std::vector<Handle> &evicted) {
  // Use the first fit strategy to evict allocated blocks.
  size = alignedSize(size, alignment_);
  bool hasSeenNonEvicted{false};
  uint64_t startAddress = 0;
  uint64_t begin = 0;
  llvm::SmallVector<std::pair<Segment, Handle>, 16> evictionCandidates;
  for (auto it = segments_.begin(), e = segments_.end(); it != e; it++) {
    // Skip any allocations below the start address.
    if (it->begin_ < startAddress) {
      continue;
    }
    auto curHandle = getHandle(it->begin_);
    if (mustNotEvict.count(curHandle)) {
      DEBUG_GLOW(llvm::dbgs()
                 << "Cannot evict a buffer from '" << name_ << "' : "
                 << "address: " << it->begin_ << " size: " << size << "\n");
      // The block cannot be evicted. Start looking after it.
      begin = it->end_;
      evictionCandidates.clear();
      hasSeenNonEvicted = true;
      continue;
    }
    // Remember current block as a candidate.
    evictionCandidates.emplace_back(std::make_pair(*it, curHandle));
    // If the total to be evicted size is enough, no need to look any further.
    if (it->end_ - begin >= size) {
      break;
    }
  }

  if ((!evictionCandidates.empty() &&
       evictionCandidates.back().first.end_ - begin >= size) ||
      (!hasSeenNonEvicted && poolSize_ >= size)) {
    // Now evict all eviction candidates.
    for (auto &candidate : evictionCandidates) {
      auto &curHandle = candidate.second;
      auto &segment = candidate.first;
      (void)segment;
      DEBUG_GLOW(llvm::dbgs() << "Evict a buffer from the '" << name_ << "': "
                              << "address: " << segment.begin_
                              << " size: " << segment.size() << "\n");
      deallocate(curHandle);
      evicted.emplace_back(curHandle);
    }
  }
}

uint64_t MemoryAllocator::allocate(uint64_t size, Handle handle,
                                   const std::set<Handle> &mustNotEvict,
                                   std::vector<Handle> &evicted) {
  // Try the usual allocation first.
  auto ptr = allocate(size, handle);
  // If it was possible to allocate the requested block, just return it.
  if (ptr != npos) {
    return ptr;
  }
  // Allocation was not possible, try to evict something.
  // Use the first fit strategy to evict allocated blocks.
  evictFirstFit(size, mustNotEvict, evicted);
  // Try again to allocate the space. This time it should succeed.
  ptr = allocate(size, handle);
  return ptr;
}

void MemoryAllocator::deallocate(Handle handle) {
  auto ptr = getAddress(handle);
  for (auto it = segments_.begin(), e = segments_.end(); it != e; it++) {
    if (it->begin_ == ptr) {
      segments_.erase(it);
      addrToHandleMap_.erase(ptr);
      handleToSegmentMap_.erase(handle);
      return;
    }
  }
  llvm_unreachable("Unknown buffer to deallocate");
}

bool MemoryAllocator::hasHandle(uint64_t address) const {
  auto it = addrToHandleMap_.find(address);
  return it != addrToHandleMap_.end();
}

MemoryAllocator::Handle MemoryAllocator::getHandle(uint64_t address) const {
  auto it = addrToHandleMap_.find(address);
  assert(it != addrToHandleMap_.end() && "Unknown address");
  return it->second;
}

bool MemoryAllocator::hasAddress(Handle handle) const {
  auto it = handleToSegmentMap_.find(handle);
  return it != handleToSegmentMap_.end();
}

Segment MemoryAllocator::getSegment(Handle handle) const {
  auto it = handleToSegmentMap_.find(handle);
  assert(it != handleToSegmentMap_.end() && "Unknown handle");
  return it->second;
}

uint64_t MemoryAllocator::getAddress(Handle handle) const {
  return getSegment(handle).begin_;
}

uint64_t MemoryAllocator::getSize(Handle handle) const {
  return getSegment(handle).size();
}

void MemoryAllocator::setHandle(uint64_t ptr, uint64_t size, Handle handle) {
  // TODO: Check that ptr is an allocated address.
  assert(contains(ptr) && "The address is not allocated");
  assert(!hasHandle(ptr) && "The address has an associated handle already");
  addrToHandleMap_[ptr] = handle;
  handleToSegmentMap_.insert(std::make_pair(handle, Segment(ptr, ptr + size)));
}

// -----------------------------------------------------------------------------
//                                 UTILS
// -----------------------------------------------------------------------------
/// Utility function to verify if two integer intervals overlap. Each interval
/// is described by a half-open interval [begin, end).
template <class T>
inline bool intervalsOverlap(T begin1, T end1, T begin2, T end2) {
  return (std::max(begin1, begin2) < std::min(end1, end2));
}

/// Utility function to verify allocations. \returns true if allocation are
/// valid and false otherwise.
bool verifyAllocations(const std::vector<Allocation> &allocArray) {

  // Allocation array length must be even.
  size_t allocNum = allocArray.size();
  if (allocNum % 2) {
    return false;
  }

  // Number of ALLOC must be equal to number of FREE.
  std::list<size_t> allocIdxList;
  for (size_t idx = 0; idx < allocNum; ++idx) {
    if (allocArray[idx].alloc_) {
      allocIdxList.push_back(idx);
    }
  }
  if (allocIdxList.size() != (allocNum / 2)) {
    return false;
  }

  // Verify each ALLOC has an associated FREE following it.
  // Verify each ALLOC has a unique handle.
  std::list<MemoryHandle> allocHandleList;
  for (auto allocIdx : allocIdxList) {
    // Find a FREE instruction associated to this ALLOC.
    auto allocHandle = allocArray[allocIdx].handle_;
    bool freeFound = false;
    for (size_t idx = allocIdx + 1; idx < allocNum; ++idx) {
      if ((!allocArray[idx].alloc_) &&
          (allocArray[idx].handle_ == allocHandle)) {
        freeFound = true;
        break;
      }
    }
    if (!freeFound) {
      return false;
    }
    // Verify ALLOC handle is unique.
    auto it =
        std::find(allocHandleList.begin(), allocHandleList.end(), allocHandle);
    if (it != allocHandleList.end()) {
      return false;
    }
    allocHandleList.push_back(allocHandle);
  }

  return true;
}

// TODO
// 1. Validate that output segments are consistent!
// 2. Add "allocate" flavors using IDs instead of Handles.
// 3. Compute and return allocation efficiency.
// 4. Remove addrToHandleMap_ from allocator.

/// \returns the total memory usage, or MemoryAllocator::npos, if the
/// allocation failed.
uint64_t MemoryAllocator::allocate(const std::vector<Allocation> &allocArray) {

  // Reset memory allocator object.
  reset();

  // Verify allocations.
  assert(verifyAllocations(allocArray) && "Allocation array invalid!");

  // If allocation array is empty then return early.
  size_t allocNum = allocArray.size();
  if (allocNum == 0) {
    return 0;
  }

  // Number of buffers/segments to allocate.
  assert((allocNum % 2 == 0) &&
         "The allocation array must have an even number of entries!");
  size_t buffNum = allocNum / 2;

  // Map Handles to consecutive unique IDs between 0 and numBuff - 1 since this
  // makes the algorithm implementation easier/faster by using IDs as vector
  // indices.
  std::unordered_map<Handle, size_t> handleToIdMap;
  std::vector<Handle> idToHandleMap(buffNum);
  size_t id = 0;
  for (const auto &alloc : allocArray) {
    // We only map the Handles of ALLOCs.
    if (alloc.alloc_) {
      idToHandleMap[id] = alloc.handle_;
      handleToIdMap[alloc.handle_] = id;
      id++;
    }
  }
  assert(id == buffNum && "Inconsistent Handle to ID mapping!");

  // -----------------------------------------------------------------------
  // Get overall information about all the buffers.
  // -----------------------------------------------------------------------
  // Array with the size for each buffer.
  std::vector<uint64_t> buffSizeArray(buffNum);

  // Array with the start/stop time (both inclusive) for each buffer.
  std::vector<uint64_t> buffTimeStart(buffNum);
  std::vector<uint64_t> buffTimeStop(buffNum);

  // The maximum total required memory of all the live buffers reached during
  // all allocation time steps. Note that this is the best size any allocation
  // algorithm can hope for and is used to compute the allocation efficiency.
  uint64_t liveSizeMax = 0;

  // Array with the total required memory of all the live buffers for each
  // allocation time step.
  std::vector<uint64_t> liveBuffSizeArray(allocNum);

  // Array with lists of IDs of all the live buffers for each allocation time
  // step.
  std::vector<std::list<uint64_t>> liveBuffIdListArray(allocNum);

  // Gather information.
  {
    uint64_t liveBuffSize = 0;
    std::list<uint64_t> liveBuffIdList;
    for (size_t allocIdx = 0; allocIdx < allocNum; allocIdx++) {

      // Current buffer handle and mapped ID.
      auto buffHandle = allocArray[allocIdx].handle_;
      auto buffId = handleToIdMap[buffHandle];

      // Current buffer size. We only use the buffer size of an ALLOC request.
      // For a FREE request we use the buffer size of the associated ALLOC.
      // We round the requested buffer size using the alignment.
      uint64_t buffSize;
      if (allocArray[allocIdx].alloc_) {
        buffSize = alignedSize(allocArray[allocIdx].size_, alignment_);
      } else {
        buffSize = buffSizeArray[buffId];
      }

      // Update buffer information.
      if (allocArray[allocIdx].alloc_) {
        buffSizeArray[buffId] = buffSize;
        buffTimeStart[buffId] = allocIdx;
      } else {
        buffTimeStop[buffId] = allocIdx;
      }

      // Update liveness information.
      if (allocArray[allocIdx].alloc_) {
        liveBuffSize = liveBuffSize + buffSize;
        liveBuffIdList.push_back(buffId);
      } else {
        liveBuffSize = liveBuffSize - buffSize;
        auto it =
            std::find(liveBuffIdList.begin(), liveBuffIdList.end(), buffId);
        assert(it != liveBuffIdList.end() &&
               "Buffer ID not found for removal!");
        liveBuffIdList.erase(it);
      }
      liveSizeMax = std::max(liveSizeMax, liveBuffSize);
      liveBuffSizeArray[allocIdx] = liveBuffSize;
      liveBuffIdListArray[allocIdx] = liveBuffIdList;
    }
    assert(liveBuffSize == 0 &&
           "Mismatch between total allocated and deallocated size!");
    assert(liveBuffIdList.empty() &&
           "Mismatch between total allocated and deallocated buffers!");
  }

  // If the theoretical required memory is larger than the available memory size
  // then we return early.
  if (poolSize_ && (liveSizeMax > poolSize_)) {
    return npos;
  }

  // ---------------------------------------------------------------------------
  // Allocate all the buffers.
  // ---------------------------------------------------------------------------
  // Local list of allocated IDs and segments.
  std::list<std::pair<size_t, Segment>> idSegList;

  // The maximum total memory used for segment allocation.
  uint64_t usedSizeMax = 0;

  // Allocate all buffers.
  for (size_t buffIdx = 0; buffIdx < buffNum; buffIdx++) {

    // -----------------------------------------------------------------------
    // Select buffer to allocate:
    // 1. Find maximum live allocation size.
    // 2. Take largest buffer from the maximum live allocation.
    // 3. If multiple buffers with same size, take the buffer with highest
    //    live interval.
    // -----------------------------------------------------------------------

    // Find maximum total live allocation.
    auto liveBuffSizeMaxIt =
        std::max_element(liveBuffSizeArray.begin(), liveBuffSizeArray.end());
    auto liveBuffSizeMaxIdx =
        std::distance(liveBuffSizeArray.begin(), liveBuffSizeMaxIt);
    auto &liveBuffIdList = liveBuffIdListArray[liveBuffSizeMaxIdx];

    // Find buffer with maximum size within the maximum allocation.
    uint64_t buffIdMax = 0;
    uint64_t buffSizeMax = 0;
    for (auto buffIdIter : liveBuffIdList) {
      // If size is the same choose buffer with maximum live interval.
      if (buffSizeArray[buffIdIter] == buffSizeMax) {
        auto currTime = buffTimeStop[buffIdMax] - buffTimeStart[buffIdMax];
        auto iterTime = buffTimeStop[buffIdIter] - buffTimeStart[buffIdIter];
        if (iterTime > currTime) {
          buffIdMax = buffIdIter;
        }
      }
      // Choose largest buffer.
      if (buffSizeArray[buffIdIter] > buffSizeMax) {
        buffSizeMax = buffSizeArray[buffIdIter];
        buffIdMax = buffIdIter;
      }
    }

    // Current segment ID and size.
    auto currSegId = buffIdMax;
    auto currSegSize = buffSizeMax;

    // -----------------------------------------------------------------------
    // Find previously allocated segments which overlap with the current segment
    // in time, that is segments which are alive at the same time with the
    // current segment. We keep only those segments and store them in buffers.
    // We also sort the found segments in increasing order of the stop address.
    // Note: The number of previous segments is usually small.
    // -----------------------------------------------------------------------
    typedef std::pair<uint64_t, uint64_t> AddressPair;

    // We initialize the "previous segments" buffers with a virtual segment of
    // size 0 since this will simplify the logic used in the following section.
    std::vector<AddressPair> prevSegAddr = {AddressPair(0, 0)};

    for (const auto &idSeg : idSegList) {

      // Previously allocated segment.
      auto prevSegId = idSeg.first;
      auto prevSeg = idSeg.second;

      // Verify if the previous segment overlaps with current segment in time.
      bool overlap =
          intervalsOverlap(buffTimeStart[currSegId], buffTimeStop[currSegId],
                           buffTimeStart[prevSegId], buffTimeStop[prevSegId]);

      // If segment overlaps with previous then store the previous segment.
      if (overlap) {
        prevSegAddr.push_back(AddressPair(prevSeg.begin_, prevSeg.end_));
      }
    }

    // Order segments in the increasing order of the stop address.
    std::sort(prevSegAddr.begin(), prevSegAddr.end(),
              [](const AddressPair &a, const AddressPair &b) {
                return a.second < b.second;
              });

    // -----------------------------------------------------------------------
    // Find a position for the current segment by trying to allocate at the
    // end of all the previously allocated segments which were previously
    // found. Since the previous segments are ordered by their stop address
    // in ascending order this procedure is guaranteed to find a place at
    // least at the end of the last segment.
    // -----------------------------------------------------------------------

    uint64_t currSegAddrStart = 0;
    uint64_t currSegAddrStop = 0;

    for (size_t prevSegIdx = 0; prevSegIdx < prevSegAddr.size(); prevSegIdx++) {

      // Try to place current segment after this previously allocated segment.
      currSegAddrStart = prevSegAddr[prevSegIdx].second;
      currSegAddrStop = currSegAddrStart + currSegSize;

      // Verify if this placement overlaps with all the other segments.
      // Note that this verification with all the previous segments is required
      // because the previous segments can overlap between themselves.
      bool overlap = false;
      for (size_t ovrSegIdx = 0; ovrSegIdx < prevSegAddr.size(); ovrSegIdx++) {
        // Check overlap.
        overlap = overlap || intervalsOverlap(currSegAddrStart, currSegAddrStop,
                                              prevSegAddr[ovrSegIdx].first,
                                              prevSegAddr[ovrSegIdx].second);
        // Early break if overlaps.
        if (overlap) {
          break;
        }
      }

      // If no overlap than we found the solution for the placement.
      if (!overlap) {
        break;
      }
    }

    // Update maximum used size.
    usedSizeMax = std::max(usedSizeMax, currSegAddrStop);

    // If max available memory is surpassed with the new segment then we stop
    // the allocation and return.
    if (poolSize_ && (usedSizeMax > poolSize_)) {
      return npos;
    }

    // Allocate current segment.
    Segment currSeg(currSegAddrStart, currSegAddrStop);
    idSegList.push_back(std::pair<size_t, Segment>(currSegId, currSeg));

    // Update buffer information.
    for (size_t allocIdx = buffTimeStart[currSegId];
         allocIdx < buffTimeStop[currSegId]; allocIdx++) {
      // Update total live sizes.
      liveBuffSizeArray[allocIdx] =
          liveBuffSizeArray[allocIdx] - buffSizeArray[currSegId];
      // Update total live IDs.
      auto &allocIds = liveBuffIdListArray[allocIdx];
      auto it = std::find(allocIds.begin(), allocIds.end(), currSegId);
      assert(it != allocIds.end() && "Buffer ID not found for removal!");
      allocIds.erase(it);
    }
  }

  // Verify again that all the buffers were allocated.
  for (size_t allocIdx = 0; allocIdx < allocNum; allocIdx++) {
    assert(liveBuffSizeArray[allocIdx] == 0 &&
           "Not all buffers were allocated!");
    assert(liveBuffIdListArray[allocIdx].empty() &&
           "Not all buffers were allocated!");
  }

  // Print statistics.
#if 0
  float allocEfficiency = (float)(liveSizeMax) / (float)(usedSizeMax);
  printf("liveSizeMax: %I64d\n", liveSizeMax);
  printf("usedSizeMax: %I64d\n", usedSizeMax);
  printf("allocEff   : %f\n", allocEfficiency);
#endif

  // Update the segments, handles and the max used memory.
  for (const auto &idSeg : idSegList) {
    size_t id = idSeg.first;
    Segment segment = idSeg.second;
    Handle handle = idToHandleMap[id];
    // Add segment.
    segments_.push_back(segment);
    // Add handle.
    handleToSegmentMap_.insert(std::make_pair(handle, segment));
  }
  maxMemoryAllocated_ = usedSizeMax;
  return usedSizeMax;
}
