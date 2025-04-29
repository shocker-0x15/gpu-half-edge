#include "gpu_shared.h"
#if !defined(__INTELLISENSE__)
#include <cub/cub.cuh>
#endif

struct DirectedEdgeLessOp {
    CUDA_DEVICE_FUNCTION CUDA_INLINE bool operator()(
        const shared::DirectedEdge &l, const shared::DirectedEdge &r) const
    {
        return l < r;
    }
};

size_t getScratchMemSizeForSortDirectedEdges(uint32_t numEdges) {
    size_t size;
    cub::DeviceMergeSort::StableSortPairs<shared::DirectedEdge*, uint32_t*>(
        nullptr, size,
        nullptr, nullptr, numEdges, DirectedEdgeLessOp());
    return size;
}

// TODO: Use radix sort?
void sortDirectedEdges(
    shared::DirectedEdge* edges, uint32_t* edgeIndices, uint32_t numEdges,
    void* scratchMem, size_t scratchMemSize)
{
    cub::DeviceMergeSort::StableSortPairs<shared::DirectedEdge*, uint32_t*>(
        scratchMem, scratchMemSize,
        edges, edgeIndices, numEdges, DirectedEdgeLessOp());
}
