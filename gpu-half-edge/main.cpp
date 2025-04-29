#include "gpu_shared.h"

#include <assimp/scene.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>

#include "../ext/cubd/cubd.h"

size_t getScratchMemSizeForSortDirectedEdges(uint32_t numEdges);

void sortDirectedEdges(
    shared::DirectedEdge* edges, uint32_t* edgeIndices, uint32_t numEdges,
    void* scratchMem, size_t scratchMemSize);



int32_t mainFunc(int32_t argc, const char* argv[]) {
    const std::filesystem::path resourceDir = getExecutableDirectory() / "gpu-half-edge";

    std::filesystem::path filePath;
    {
        uint32_t argIdx = 1;
        while (argIdx < argc) {
            std::string_view arg = argv[argIdx];
            if (!arg.starts_with("--")) {
                hpprintf("Invalid argument.\n");
                return -1;
            }

            const auto checkArgEnd = [&]
            (const std::string_view &param, int32_t lastArgIdx) {
                if (lastArgIdx >= argc) {
                    hpprintf("Insufficient number of arguments for %s\n", param.data());
                    return false;
                }
                return true;
            };

            arg = std::string_view(argv[argIdx] + 2);
            if (arg == "mesh") {
                if (!checkArgEnd(arg, argIdx + 1))
                    return -1;
                filePath = argv[argIdx + 1];
                argIdx += 2;
            }
            else {
                hpprintf("Unkown argument: --%s.\n", arg.data());
                return -1;
            }
        }
    }



    // JP: メッシュの読み込み。
    // EN: Load a mesh.
    hpprintf("Reading: %s ... ", filePath.string().c_str());
    fflush(stdout);
    Assimp::Importer importer;
    const aiScene* const srcScene = importer.ReadFile(
        filePath.string(),
        aiProcess_JoinIdenticalVertices);
    if (!srcScene) {
        hpprintf("Failed to load %s.\n", filePath.string().c_str());
        return -1;
    }
    hpprintf("done.\n");



    CUcontext cuContext;
    CUstream cuStream;
    CUDADRV_CHECK(cuInit(0));
    CUDADRV_CHECK(cuCtxCreate(&cuContext, 0, 0));
    CUDADRV_CHECK(cuCtxSetCurrent(cuContext));
    CUDADRV_CHECK(cuStreamCreate(&cuStream, 0));

    const std::filesystem::path ptxDir = resourceDir / "ptxes";

    CUmodule cuModule;
    CUDADRV_CHECK(cuModuleLoad(
        &cuModule,
        (ptxDir / "gpu_kernels.ptx").string().c_str()));

    cudau::Kernel initializeHalfEdges(
        cuModule, "initializeHalfEdges", cudau::dim3(32, 1, 1), 0);
    cudau::Kernel findTwinHalfEdges(
        cuModule, "findTwinHalfEdges", cudau::dim3(32, 1, 1), 0);
    cudau::Kernel findAdjacentFaces(
        cuModule, "findAdjacentFaces", cudau::dim3(32, 1, 1), 0);



    constexpr cudau::BufferType cudaBufferType = cudau::BufferType::Device;

    // JP: メッシュごとにハーフエッジデータ構造を構築する。
    // EN: Construct a half-edge data structure for each mesh.
    for (uint32_t meshIdx = 0; meshIdx < srcScene->mNumMeshes; ++meshIdx) {
        const aiMesh* srcMesh = srcScene->mMeshes[meshIdx];
        const uint32_t faceCount = srcMesh->mNumFaces;

        // JP: 面ごとの頂点インデックス列を詰めたリストと、面ごとのエッジ数リストを作る。
        // EN: Create a packed list of vertex indices for each face and a list of the number of edges per face.
        std::vector<uint32_t> vtxIndicesOnHost;
        std::vector<uint32_t> faceEdgeCountsOnHost(faceCount + 1);
        for (uint32_t faceIdx = 0; faceIdx < faceCount; ++faceIdx) {
            const aiFace &srcFace = srcMesh->mFaces[faceIdx];

            for (uint32_t f_vIdx = 0; f_vIdx < srcFace.mNumIndices; ++f_vIdx) {
                vtxIndicesOnHost.push_back(srcFace.mIndices[f_vIdx]);
            }
            faceEdgeCountsOnHost[faceIdx] = srcFace.mNumIndices;
        }
        faceEdgeCountsOnHost[faceCount] = 0; // sentinel
        const uint32_t halfEdgeCount = vtxIndicesOnHost.size();



        // JP: 必要なスクラッチメモリサイズを計算する。
        // EN: Calculate the required scratch memory size.
        size_t scratchSizeForCommonOps = 0;
        {
            size_t scratchSize;
            cubd::DeviceScan::ExclusiveSum<uint32_t*, uint32_t*>(
                nullptr, scratchSize, nullptr, nullptr, faceCount + 1);
            scratchSizeForCommonOps = std::max(scratchSize, scratchSizeForCommonOps);

            scratchSize = getScratchMemSizeForSortDirectedEdges(halfEdgeCount);
            scratchSizeForCommonOps = std::max(scratchSize, scratchSizeForCommonOps);
        }

        cudau::Buffer scratchMemForCommonOps(
            cuContext, cudaBufferType, scratchSizeForCommonOps, 1);

        cudau::TypedBuffer<uint32_t> vtxIndices(
            cuContext, cudaBufferType, vtxIndicesOnHost);
        cudau::TypedBuffer<uint32_t> faceEdgeCounts(
            cuContext, cudaBufferType, faceEdgeCountsOnHost);
        cudau::TypedBuffer<shared::DirectedEdge> directedEdges(
            cuContext, cudaBufferType, halfEdgeCount);
        cudau::TypedBuffer<uint32_t> halfEdgeIndices(
            cuContext, cudaBufferType, halfEdgeCount);
        cudau::TypedBuffer<shared::HalfEdge> halfEdges(
            cuContext, cudaBufferType, halfEdgeCount);

        cudau::TypedBuffer<uint32_t> adjFaceIndices(
            cuContext, cudaBufferType, halfEdgeCount);



        // ----------------------------------------------------------------
        // JP: ハーフエッジデータ構造を構築する。
        // EN: Construct a half-edge data structure.

        size_t scratchSize;

        // JP: 面ごとのエッジ数リストをスキャンして、ハーフエッジベースインデックスリストを得る。
        //     まぁこれはホスト側ですでに求まっているかもしれない。
        // EN: Scan the face edge count list to get the half-edge base index list.
        //     Well, this could be already available on the host.
        scratchSize = scratchMemForCommonOps.sizeInBytes();
        cubd::DeviceScan::ExclusiveSum(
            scratchMemForCommonOps.getDevicePointer(), scratchSize,
            faceEdgeCounts.getDevicePointer(), faceEdgeCounts.getDevicePointer(),
            faceCount + 1, cuStream);

        //std::vector<uint32_t> faceOffsetsOnHost = faceEdgeCounts;

        // JP: ハーフエッジ構造を初期化する。
        // EN: Initialize half-edge data structures.
        initializeHalfEdges.launchWithThreadDim(
            cuStream, cudau::dim3(faceCount),
            vtxIndices, faceEdgeCounts, faceCount,
            directedEdges, halfEdgeIndices, halfEdges);

        //std::vector<shared::DirectedEdge> directedEdgesOnHost = directedEdges;
        //std::vector<uint32_t> halfEdgeIndicesOnHost = halfEdgeIndices;
        //std::vector<shared::HalfEdge> halfEdgesOnHost = halfEdges;

        // JP: エッジとハーフエッジインデックスの配列をソートする。
        // EN: Sort the arrays of edges and half edge indices.
        sortDirectedEdges(
            directedEdges.getDevicePointer(), halfEdgeIndices.getDevicePointer(), halfEdgeCount,
            scratchMemForCommonOps.getDevicePointer(), scratchMemForCommonOps.sizeInBytes());

        //std::vector<shared::DirectedEdge> directedEdgesOnHost = directedEdges;
        //std::vector<uint32_t> halfEdgeIndicesOnHost = halfEdgeIndices;

        // JP: 双子のハーフエッジを特定する。
        // EN: Find the twin for each half edge.
        findTwinHalfEdges.launchWithThreadDim(
            cuStream, cudau::dim3(halfEdgeCount),
            directedEdges, halfEdgeIndices, halfEdgeCount,
            halfEdges);

        // END: Construct a half-edge data structure.
        // ----------------------------------------------------------------



        // JP: ハーフエッジデータ構造を用いて各面の隣接面リストを得る。
        // EN: Obtain the adjacent face list for each face using the 
        //     half-edge data structure.
        findAdjacentFaces.launchWithThreadDim(
            cuStream, cudau::dim3(faceCount),
            faceEdgeCounts, faceCount, halfEdges,
            adjFaceIndices);

        // JP: ホストに結果を読み出して表示。
        // EN: Read back the result to the host and print it.
        printf("Mesh %u:\n", meshIdx);
        std::vector<uint32_t> faceOffsetsOnHost = faceEdgeCounts;
        std::vector<uint32_t> adjFaceIndicesOnHost = adjFaceIndices;
        for (uint32_t faceIdx = 0; faceIdx < faceCount; ++faceIdx) {
            const uint32_t faceOffset = faceOffsetsOnHost[faceIdx];
            const uint32_t nextFaceOffset = faceOffsetsOnHost[faceIdx + 1];
            const uint32_t faceEdgeCount = nextFaceOffset - faceOffset;

            printf("%u: %u edges, (", faceIdx, faceEdgeCount);
            uint32_t adjCount = 0;
            for (uint32_t f_eIdx = 0; f_eIdx < faceEdgeCount; ++f_eIdx) {
                const uint32_t nbFaceIdx = adjFaceIndicesOnHost[faceOffset + f_eIdx];
                if (nbFaceIdx != 0xFFFF'FFFF) {
                    printf("%u, ", nbFaceIdx);
                    ++adjCount;
                }
            }
            printf("), %u adjs\n", adjCount);
        }
        printf("\n");
    }

    return 0;
}

int32_t main(int32_t argc, const char* argv[]) {
    try {
        return mainFunc(argc, argv);
    }
    catch (const std::exception &ex) {
        hpprintf("Error: %s\n", ex.what());
        return -1;
    }
}
