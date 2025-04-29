#include "gpu_shared.h"

using namespace shared;



CUDA_DEVICE_KERNEL void initializeHalfEdges(
    const uint32_t* const vtxIndices, const uint32_t* const faceOffsets, const uint32_t faceCount,
    DirectedEdge* const edges, uint32_t* const halfEdgeIndices, HalfEdge* const halfEdges)
{
    const uint32_t faceIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (faceIdx >= faceCount)
        return;

    const uint32_t faceOffset = faceOffsets[faceIdx];
    const uint32_t nextFaceOffset = faceOffsets[faceIdx + 1];
    const uint32_t faceEdgeCount = nextFaceOffset - faceOffset;

    for (uint32_t f_eIdx = 0; f_eIdx < faceEdgeCount; ++f_eIdx) {
        const uint32_t edgeIdx = faceOffset + f_eIdx;
        const uint32_t nextEdgeIdx = faceOffset + (f_eIdx + 1) % faceEdgeCount;
        const uint32_t prevEdgeIdx = faceOffset + (f_eIdx + (faceEdgeCount - 1)) % faceEdgeCount;

        const uint32_t vIdx = vtxIndices[edgeIdx];

        DirectedEdge &edge = edges[edgeIdx];
        edge.vertexIndexA = vIdx;
        edge.vertexIndexB = vtxIndices[nextEdgeIdx];

        halfEdgeIndices[edgeIdx] = edgeIdx;

        HalfEdge &halfEdge = halfEdges[edgeIdx];
        halfEdge.twinHalfEdgeIndex = 0xFFFF'FFFF;
        halfEdge.orgVertexIndex = vIdx;
        halfEdge.faceIndex = faceIdx;
        halfEdge.prevHalfEdgeIndex = prevEdgeIdx;
        halfEdge.nextHalfEdgeIndex = nextEdgeIdx;
    }
}

CUDA_DEVICE_KERNEL void findTwinHalfEdges(
    const DirectedEdge* const sortedEdges, const uint32_t* const sortedHalfEdgeIndices, const uint32_t halfEdgeCount,
    const uint32_t* const faceOffsets, HalfEdge* const halfEdges)
{
    const uint32_t halfEdgeIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (halfEdgeIdx >= halfEdgeCount)
        return;

    HalfEdge &halfEdge = halfEdges[halfEdgeIdx];
    const uint32_t faceOffset = faceOffsets[halfEdge.faceIndex];
    const uint32_t nextFaceOffset = faceOffsets[halfEdge.faceIndex + 1];
    const uint32_t faceEdgeCount = nextFaceOffset - faceOffset;
    const uint32_t nextHalfEdgeIdx = faceOffset + ((halfEdgeIdx - faceOffset) + 1) % faceEdgeCount;
    const HalfEdge &nextHalfEdge = halfEdges[nextHalfEdgeIdx];

    DirectedEdge twinEdge;
    twinEdge.vertexIndexA = nextHalfEdge.orgVertexIndex;
    twinEdge.vertexIndexB = halfEdge.orgVertexIndex;
    uint32_t twinHalfEdgeIdx = 0;
    bool found = false;
    for (uint32_t d = nextPowOf2(halfEdgeCount) >> 1; d >= 1; d >>= 1) {
        if (twinHalfEdgeIdx + d >= halfEdgeCount)
            continue;
        if (sortedEdges[twinHalfEdgeIdx + d] <= twinEdge) {
            twinHalfEdgeIdx += d;
            found = sortedEdges[twinHalfEdgeIdx] == twinEdge;
            if (found)
                break;
        }
    }

    if (found)
        halfEdge.twinHalfEdgeIndex = sortedHalfEdgeIndices[twinHalfEdgeIdx];
}



CUDA_DEVICE_KERNEL void findNeighborFaces(
    const uint32_t* const faceOffsets, const uint32_t faceCount, const HalfEdge* const halfEdges,
    uint32_t* const neighborFaceIndices)
{
    const uint32_t faceIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (faceIdx >= faceCount)
        return;

    const uint32_t faceOffset = faceOffsets[faceIdx];
    const uint32_t nextFaceOffset = faceOffsets[faceIdx + 1];
    const uint32_t faceEdgeCount = nextFaceOffset - faceOffset;
    for (uint32_t f_eIdx = 0; f_eIdx < faceEdgeCount; ++f_eIdx) {
        const uint32_t halfEdgeIdx = faceOffset + f_eIdx;
        const HalfEdge &halfEdge = halfEdges[halfEdgeIdx];
        uint32_t nbFaceIdx = 0xFFFF'FFFF;
        if (halfEdge.twinHalfEdgeIndex != 0xFFFF'FFFF) {
            const HalfEdge &twinHalfEdge = halfEdges[halfEdge.twinHalfEdgeIndex];
            nbFaceIdx = twinHalfEdge.faceIndex;
        }
        neighborFaceIndices[faceOffset + f_eIdx] = nbFaceIdx;
    }
}
