/*

   Copyright 2024 Shin Watanabe

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

*/

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
    HalfEdge* const halfEdges)
{
    const uint32_t halfEdgeIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (halfEdgeIdx >= halfEdgeCount)
        return;

    // JP: 同じ頂点インデックスペア(ただし逆向き)を持つエッジを作成する。
    // EN: Create an edge with the same vertex index pair (but reversed).
    HalfEdge &halfEdge = halfEdges[halfEdgeIdx];
    const HalfEdge &nextHalfEdge = halfEdges[halfEdge.nextHalfEdgeIndex];
    DirectedEdge twinEdge;
    twinEdge.vertexIndexA = nextHalfEdge.orgVertexIndex;
    twinEdge.vertexIndexB = halfEdge.orgVertexIndex;

    // JP: ソートされたエッジ列を二分探索して作成したエッジに対応するものを見つける。
    // EN: Binary search the sorted edges to find the corresponding edge for the one created.
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

    // JP: 見つかったハーフエッジを双子として登録する。
    // EN: Register the found half edge as the twin.
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
    // JP: 各辺のハーフエッジの双子から隣接面を特定する。
    // EN: Identify a neighboring face using the twin of each half edge.
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
