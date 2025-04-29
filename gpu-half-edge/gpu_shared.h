#pragma once

#include "../common/common.h"

namespace shared {

struct DirectedEdge {
    uint32_t vertexIndexA;
    uint32_t vertexIndexB;

    CUDA_DEVICE_FUNCTION bool operator==(const DirectedEdge &r) const {
        return vertexIndexA == r.vertexIndexA && vertexIndexB == r.vertexIndexB;
    }
    CUDA_DEVICE_FUNCTION bool operator!=(const DirectedEdge &r) const {
        return vertexIndexA != r.vertexIndexA || vertexIndexB != r.vertexIndexB;
    }

    CUDA_DEVICE_FUNCTION bool operator<(const DirectedEdge &r) const {
        if (vertexIndexA < r.vertexIndexA)
            return true;
        if (vertexIndexA > r.vertexIndexA)
            return false;

        // vertexIndexA == r.vertexIndexA
        if (vertexIndexB < r.vertexIndexB)
            return true;
        if (vertexIndexB > r.vertexIndexB)
            return false;

        // *this == r
        return false;
    }
    CUDA_DEVICE_FUNCTION bool operator<=(const DirectedEdge &r) const {
        if (vertexIndexA < r.vertexIndexA)
            return true;
        if (vertexIndexA > r.vertexIndexA)
            return false;

        // vertexIndexA == r.vertexIndexA
        if (vertexIndexB < r.vertexIndexB)
            return true;
        if (vertexIndexB > r.vertexIndexB)
            return false;

        // *this == r
        return true;
    }
};

struct HalfEdge {
    uint32_t twinHalfEdgeIndex;
    uint32_t orgVertexIndex;
    uint32_t faceIndex;
    uint32_t prevHalfEdgeIndex;
    uint32_t nextHalfEdgeIndex;
};

}
