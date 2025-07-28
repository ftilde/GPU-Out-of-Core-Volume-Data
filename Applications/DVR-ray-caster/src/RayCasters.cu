#include <RayCasters_helpers.hpp>

// CUDA math helper
#include <helper_math.h>

namespace tdns
{
namespace graphics
{

    //__device__ inline float stepToBrickEnd(float3 position, float3 direction, float3 brickSize) {
    //    float3 inBrickPos = fmodf(position + make_float3(1.0f, 1.0f, 1.0f), brickSize);
    //    float3 spacePlus = brickSize - inBrickPos;
    //    float3 spaceMinus = -inBrickPos;
    //    float3 stepsPlus = spacePlus/direction;
    //    float3 stepsMinus = spaceMinus/direction;

    //    float steps[6] = {stepsPlus.x, stepsPlus.y, stepsPlus.z, stepsMinus.x, stepsMinus.y, stepsMinus.z};

    //    float minStep=2.0*brickSize.x;
    //    for(int i=0; i<6; ++i) {
    //        float step = steps[i];
    //        if(step > 0) {
    //            minStep = min(minStep, step);
    //        }
    //    }

    //    if(minStep == 2.0*brickSize.x) {
    //        return brickSize.x;
    //    }

    //    return minStep;
    //}

    template<typename T>
    __device__ void RayCastDVRImpl(uint32_t *pixelBuffer,
                            cudaTextureObject_t tfTex,
                            uint2 screenSize,
                            uint32_t renderScreenWidth,
                            float fov,
                            float3 bboxMin, float3 bboxMax,
                            int32_t steps, float tstep,
                            tdns::gpucache::K_CacheManager<T> manager,
                            float3 *invLevelsSize,
                            uint3 *levelsSize,
                            float3 *LODBrickSize,
                            float *LODStepSize,
                            uint32_t numLODs,
                            size_t seed)
    {
        // 2D Thread ID
        uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

        uint32_t xNeighbor = x+1;
        if (xNeighbor > screenSize.x) {
            xNeighbor = x-1;
        }
        uint32_t yNeighbor = y+1;
        if (yNeighbor > screenSize.y) {
            yNeighbor = y-1;
        }

        if (x > screenSize.x || y > screenSize.y) return;

        // Transform the 2D screen coords into [-1; 1] range
        float u = ((x + 0.5f) / (float) screenSize.x) * 2.0f - 1.0f;
        float uNeighbor = ((xNeighbor + 0.5f) / (float) screenSize.x) * 2.0f - 1.0f;
        float v = ((y + 0.5f) / (float) screenSize.y) * 2.0f - 1.0f;
        float vNeighbor = ((yNeighbor + 0.5f) / (float) screenSize.y) * 2.0f - 1.0f;

        // calculate eye ray in world space
        float3 origin = make_float3(mul(d_invModelViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
        float3 direction = normalize(make_float3(u, v, -2.f));
        direction = mul(d_invModelViewMatrix, direction);
        float3 directionXN = normalize(make_float3(uNeighbor, v, -2.f));
        directionXN = mul(d_invModelViewMatrix, directionXN);
        float3 directionYN = normalize(make_float3(u, vNeighbor, -2.f));
        directionYN = mul(d_invModelViewMatrix, directionYN);

        float3 directionLeft = normalize(cross(direction, directionYN));
        float3 directionUp = normalize(cross(direction, directionXN));

        // find intersection with box
        float tnear, tfar;
        int32_t intersected = intersectBox(origin, direction, bboxMin, bboxMax, &tnear, &tfar);
        
        const float3 bgColor = {0.f, 0.f, 0.f};
        float4 finalColor = {0.f, 0.f, 0.f, 0.f};

        if (!intersected)
        {
            finalColor.x = bgColor.x;
            finalColor.y = bgColor.y;
            finalColor.z = bgColor.z;
        }
        else
        {
            if (tnear < 0.f) tnear = 0.f;          // clamp to near plane

            float t = tnear;
            float3 position = origin + direction * tnear;
            float3 step;
            // float3 step = direction * tstep;
            float sample;
            float4 color = {0.f, 0.f, 0.f, 0.f};
            float3 texturePosition = {0.f, 0.f, 0.f};
            float3 normalizedPosition = {0.f, 0.f, 0.f};

            float endPixelDistX = abs(dot(direction - directionXN, directionLeft) * tfar);
            float endPixelDistY = abs(dot(direction - directionYN, directionUp) * tfar);

            // Print the bouding box edges in red
            // print_bb_edges(bboxMin, bboxMax, position, finalColor);
            
            // march along ray from front to back, accumulating color
            while (finalColor.w < 0.95f && t < tfar)
            {
                float alpha = t / tfar;
                float pixelDistX = endPixelDistX * alpha;
                float pixelDistY = endPixelDistY * alpha;

                // Transform the [-1; 1] world position into a [0; 0.99[ range volume coords for the texture.
                texturePosition = position * 0.495f + 0.495f;
                texturePosition.z = 1 - texturePosition.z;

                // sampling
                normalizedPosition = clamp(texturePosition, 0.f, 0.99f);
                uint32_t lod = compute_LOD(pixelDistX, pixelDistY, directionLeft, directionUp, invLevelsSize, numLODs);
                tdns::gpucache::VoxelStatus voxelStatus = manager.template get_normalized<float>(lod, normalizedPosition, sample);

                // Handle Unmapped and Empty bricks
                if (voxelStatus == tdns::gpucache::VoxelStatus::Empty || voxelStatus == tdns::gpucache::VoxelStatus::Unmapped)
                {
                    // Welp, I tried. This still cuts of part of the volume
                    // since position%brickSize does not quite match up with the
                    // actual bricks...
                    // float3 brickSize = LODBrickSize[lod];
                    // float empty_step = max(stepToBrickEnd(position, direction, brickSize), tstep);

                    float empty_step = tstep;
                    t += empty_step;
                    if (t > tfar)
                        break;
                    position += direction * empty_step;
                    continue;
                }

                // classification
                color = tex1D<float4>(tfTex, sample);

                color.w = 1.f - powf(1.f - color.w, LODStepSize[lod] / tstep);
                // color.w = 1.f - powf(1.f - color.w, tstep / tstep);

                // skip transparent samples
                if (color.w > 0.001f)
                {
                    // pre-multiply alpha
                    color.x *= color.w;
                    color.y *= color.w;
                    color.z *= color.w;

                    // front-to-back blending operator
                    finalColor += color * (1.0f - finalColor.w);
                }

                // add the step size according to the LOD of the current sample
                t += LODStepSize[lod];
                // t += tstep;
                
                // use the step size according to the LOD of the current sample
                step = direction * LODStepSize[lod];
                // step = direction * tstep;
                position += step;
                    
            }
        }
        // Brightness
        // finalColor *= 1.5f;

        uint32_t i = y * screenSize.x + x;
        pixelBuffer[i] = rgbaFloatToInt(finalColor);
    }

    template<typename T>
    __device__ void RayCastMOPImpl(uint32_t *pixelBuffer,
                            cudaTextureObject_t tfTex,
                            uint2 screenSize,
                            uint32_t renderScreenWidth,
                            float fov,
                            float3 bboxMin, float3 bboxMax,
                            int32_t steps, float tstep,
                            tdns::gpucache::K_CacheManager<T> manager,
                            float3 *invLevelsSize,
                            uint3 *levelsSize,
                            float3 *LODBrickSize,
                            float *LODStepSize,
                            uint32_t numLODs,
                            size_t seed)
    {
        // 2D Thread ID
        uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x > screenSize.x || y > screenSize.y) return;

        uint32_t xNeighbor = x+1;
        if (xNeighbor > screenSize.x) {
            xNeighbor = x-1;
        }
        uint32_t yNeighbor = y+1;
        if (yNeighbor > screenSize.y) {
            yNeighbor = y-1;
        }

        // Transform the 2D screen coords into [-1; 1] range
        float u = ((x + 0.5f) / (float) screenSize.x) * 2.0f - 1.0f;
        float uNeighbor = ((xNeighbor + 0.5f) / (float) screenSize.x) * 2.0f - 1.0f;
        float v = ((y + 0.5f) / (float) screenSize.y) * 2.0f - 1.0f;
        float vNeighbor = ((yNeighbor + 0.5f) / (float) screenSize.y) * 2.0f - 1.0f;

        // calculate eye ray in world space
        float3 origin = make_float3(mul(d_invModelViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
        float3 direction = normalize(make_float3(u, v, -2.f));
        direction = mul(d_invModelViewMatrix, direction);
        float3 directionXN = normalize(make_float3(uNeighbor, v, -2.f));
        directionXN = mul(d_invModelViewMatrix, directionXN);
        float3 directionYN = normalize(make_float3(u, vNeighbor, -2.f));
        directionYN = mul(d_invModelViewMatrix, directionYN);

        float3 directionLeft = normalize(cross(direction, directionYN));
        float3 directionUp = normalize(cross(direction, directionXN));

        // find intersection with box
        float tnear, tfar;
        int32_t intersected = intersectBox(origin, direction, bboxMin, bboxMax, &tnear, &tfar);
        
        const float3 bgColor = {0.f, 0.f, 0.f};
        float4 finalColor = {0.f, 0.f, 0.f, 0.f};

        if (!intersected)
        {
            finalColor.x = bgColor.x;
            finalColor.y = bgColor.y;
            finalColor.z = bgColor.z;
        }
        else
        {
            if (tnear < 0.f) tnear = 0.f;          // clamp to near plane

            float t = tnear;
            float3 position = origin + direction * tnear;
            float3 step;
            // float3 step = direction * tstep;
            float maxOpacity = 0.0;
            float sample;
            float3 texturePosition = {0.f, 0.f, 0.f};
            float3 normalizedPosition = {0.f, 0.f, 0.f};

            float endPixelDistX = abs(dot(direction - directionXN, directionLeft) * tfar);
            float endPixelDistY = abs(dot(direction - directionYN, directionUp) * tfar);

            // Print the bouding box edges in red
            // print_bb_edges(bboxMin, bboxMax, position, finalColor);
            
            // march along ray from front to back, accumulating color
            while (t < tfar)
            {
                float alpha = t / tfar;
                float pixelDistX = endPixelDistX * alpha;
                float pixelDistY = endPixelDistY * alpha;

                // Transform the [-1; 1] world position into a [0; 0.99[ range volume coords for the texture.
                texturePosition = position * 0.495f + 0.495f;
                texturePosition.z = 1 - texturePosition.z;

                // sampling
                normalizedPosition = clamp(texturePosition, 0.f, 0.99f);
                uint32_t lod = compute_LOD(pixelDistX, pixelDistY, directionLeft, directionUp, invLevelsSize, numLODs);
                tdns::gpucache::VoxelStatus voxelStatus = manager.template get_normalized<float>(lod, normalizedPosition, sample);

                // Handle Unmapped and Empty bricks
                if (voxelStatus == tdns::gpucache::VoxelStatus::Empty || voxelStatus == tdns::gpucache::VoxelStatus::Unmapped)
                {
                    //float3 brickSize = LODBrickSize[lod];
                    float empty_step = tstep;
                    t += empty_step;
                    if (t > tfar)
                        break;
                    position += direction * empty_step;
                    continue;
                }

                maxOpacity = max(maxOpacity, sample);

                // color.w = 1.f - powf(1.f - color.w, tstep / tstep);

                // add the step size according to the LOD of the current sample
                t += LODStepSize[lod];
                // t += tstep;
                
                // use the step size according to the LOD of the current sample
                step = direction * LODStepSize[lod];
                // step = direction * tstep;
                position += step;
                    
            }

            // classification
            finalColor = tex1D<float4>(tfTex, maxOpacity);
        }

        // Brightness
        // finalColor *= 1.5f;

        uint32_t i = y * screenSize.x + x;
        pixelBuffer[i] = rgbaFloatToInt(finalColor);
    }

    /// DVR ------------------------------------------------------------------------
    __global__ void RayCastDVR(uint32_t *pixelBuffer,
                            cudaTextureObject_t tfTex,
                            uint2 screenSize,
                            uint32_t renderScreenWidth,
                            float fov,
                            float3 bboxMin, float3 bboxMax,
                            int32_t steps, float tstep,
                            tdns::gpucache::K_CacheManager<uchar1> manager,
                            float3 *invLevelsSize,
                            uint3 *levelsSize,
                            float3 *LODBrickSize,
                            float *LODStepSize,
                            uint32_t numLODs,
                            size_t seed)
    {
        RayCastDVRImpl(pixelBuffer, tfTex, screenSize, renderScreenWidth, fov, bboxMin, bboxMax, steps, tstep, manager, invLevelsSize, levelsSize, LODBrickSize, LODStepSize, numLODs, seed);
    }
    __global__ void RayCastDVR(uint32_t *pixelBuffer,
                            cudaTextureObject_t tfTex,
                            uint2 screenSize,
                            uint32_t renderScreenWidth,
                            float fov,
                            float3 bboxMin, float3 bboxMax,
                            int32_t steps, float tstep,
                            tdns::gpucache::K_CacheManager<ushort1> manager,
                            float3 *invLevelsSize,
                            uint3 *levelsSize,
                            float3 *LODBrickSize,
                            float *LODStepSize,
                            uint32_t numLODs,
                            size_t seed)
    {
        RayCastDVRImpl(pixelBuffer, tfTex, screenSize, renderScreenWidth, fov, bboxMin, bboxMax, steps, tstep, manager, invLevelsSize, levelsSize, LODBrickSize, LODStepSize, numLODs, seed);
    }
    __global__ void RayCastDVR(uint32_t *pixelBuffer,
                            cudaTextureObject_t tfTex,
                            uint2 screenSize,
                            uint32_t renderScreenWidth,
                            float fov,
                            float3 bboxMin, float3 bboxMax,
                            int32_t steps, float tstep,
                            tdns::gpucache::K_CacheManager<float1> manager,
                            float3 *invLevelsSize,
                            uint3 *levelsSize,
                            float3 *LODBrickSize,
                            float *LODStepSize,
                            uint32_t numLODs,
                            size_t seed)
    {
        RayCastDVRImpl(pixelBuffer, tfTex, screenSize, renderScreenWidth, fov, bboxMin, bboxMax, steps, tstep, manager, invLevelsSize, levelsSize, LODBrickSize, LODStepSize, numLODs, seed);
    }

    /// MOP ------------------------------------------------------------------------

    __global__ void RayCastMOP(uint32_t *pixelBuffer,
                            cudaTextureObject_t tfTex,
                            uint2 screenSize,
                            uint32_t renderScreenWidth,
                            float fov,
                            float3 bboxMin, float3 bboxMax,
                            int32_t steps, float tstep,
                            tdns::gpucache::K_CacheManager<uchar1> manager,
                            float3 *invLevelsSize,
                            uint3 *levelsSize,
                            float3 *LODBrickSize,
                            float *LODStepSize,
                            uint32_t numLODs,
                            size_t seed)
    {
        RayCastMOPImpl(pixelBuffer, tfTex, screenSize, renderScreenWidth, fov, bboxMin, bboxMax, steps, tstep, manager, invLevelsSize, levelsSize, LODBrickSize, LODStepSize, numLODs, seed);
    }
    __global__ void RayCastMOP(uint32_t *pixelBuffer,
                            cudaTextureObject_t tfTex,
                            uint2 screenSize,
                            uint32_t renderScreenWidth,
                            float fov,
                            float3 bboxMin, float3 bboxMax,
                            int32_t steps, float tstep,
                            tdns::gpucache::K_CacheManager<ushort1> manager,
                            float3 *invLevelsSize,
                            uint3 *levelsSize,
                            float3 *LODBrickSize,
                            float *LODStepSize,
                            uint32_t numLODs,
                            size_t seed)
    {
        RayCastMOPImpl(pixelBuffer, tfTex, screenSize, renderScreenWidth, fov, bboxMin, bboxMax, steps, tstep, manager, invLevelsSize, levelsSize, LODBrickSize, LODStepSize, numLODs, seed);
    }
    __global__ void RayCastMOP(uint32_t *pixelBuffer,
                            cudaTextureObject_t tfTex,
                            uint2 screenSize,
                            uint32_t renderScreenWidth,
                            float fov,
                            float3 bboxMin, float3 bboxMax,
                            int32_t steps, float tstep,
                            tdns::gpucache::K_CacheManager<float1> manager,
                            float3 *invLevelsSize,
                            uint3 *levelsSize,
                            float3 *LODBrickSize,
                            float *LODStepSize,
                            uint32_t numLODs,
                            size_t seed)
    {
        RayCastMOPImpl(pixelBuffer, tfTex, screenSize, renderScreenWidth, fov, bboxMin, bboxMax, steps, tstep, manager, invLevelsSize, levelsSize, LODBrickSize, LODStepSize, numLODs, seed);
    }

    // DEVICE FUNC

    //---------------------------------------------------------------------------------------------
    inline __device__ float3 operator*(float* matrix, float3 v)
    {
        float x = matrix[0] * v.x + matrix[1] * v.y + matrix[2] * v.z;
        float y = matrix[4] * v.x + matrix[5] * v.y + matrix[6] * v.z;
        float z = matrix[8] * v.x + matrix[9] * v.y + matrix[10] * v.z;

        return make_float3(x, y, z);
    }

    //---------------------------------------------------------------------------------------------
    // transform vector by matrix (no translation)
    inline __device__ float3 mul(const float4x3 &M, const float3 &v)
    {
        float3 r;
        r.x = dot(v, make_float3(M.m[0]));
        r.y = dot(v, make_float3(M.m[1]));
        r.z = dot(v, make_float3(M.m[2]));
        return r;
    }

    //---------------------------------------------------------------------------------------------
    // transform vector by matrix with translation
    inline __device__ float4 mul(const float4x3 &M, const float4 &v)
    {
        float4 r;
        r.x = dot(v, M.m[0]);
        r.y = dot(v, M.m[1]);
        r.z = dot(v, M.m[2]);
        r.w = 1.0f;
        return r;
    }

    //---------------------------------------------------------------------------------------------
    inline __device__ int intersectBox(float3 p, float3 d, float3 boxmin, float3 boxmax, float *tnear, float *tfar)
    {
        // compute intersection of ray with all six bbox planes
        float3 invR = make_float3(1.0f) / d;
        float3 tbot = invR * (boxmin - p);
        float3 ttop = invR * (boxmax - p);
    
        // re-order intersections to find smallest and largest on each axis
        float3 tmin = fminf(ttop, tbot);
        float3 tmax = fmaxf(ttop, tbot);
    
        // find the largest tmin and the smallest tmax
        float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
        float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));
    
        *tnear = largest_tmin;
        *tfar = smallest_tmax;
    
        return smallest_tmax > largest_tmin;
    }

    //---------------------------------------------------------------------------------------------
    inline __device__ uint32_t rgbaFloatToInt(float4 &rgba)
    {
        rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
        rgba.y = __saturatef(rgba.y);
        rgba.z = __saturatef(rgba.z);
        rgba.w = __saturatef(rgba.w);

        return (uint(rgba.w * 255) << 24) | (uint(rgba.z * 255) << 16) | (uint(rgba.y * 255) << 8) | uint(rgba.x * 255);
    }

    //---------------------------------------------------------------------------------------------
    inline __device__ uint32_t compute_LOD(float pixelDistX, float pixelDistY, float3 dirLeft, float3 dirUp, float3 *invLevelsSize, uint32_t numLODs)
    {
        float lod_coarseness = 1.0;

        int32_t i;
        for(i=0;i<numLODs-1; ++i) {
            int32_t next = i+1;
            float3 next_spacing = invLevelsSize[next];
            float spacingDistLeft = length(dirLeft * next_spacing);
            float spacingDistUp = length(dirUp * next_spacing);

            if(spacingDistLeft >= pixelDistX * lod_coarseness || spacingDistUp >= pixelDistY * lod_coarseness) {
                break;
            }
        }
        return i;
    }

    //---------------------------------------------------------------------------------------------
    inline __device__ void print_bb_edges(const float3 &bboxMin, const float3 &bboxMax, const float3 &position, float4 &color)
    {
        // The epsilon determine the thickness of the printed edges. The higher it is, the thicker the line will be.
        float epsilon = 0.02f;
        // Print the bouding box edges in red
        if ((position.x < (bboxMin.x + epsilon) || position.x > (bboxMax.x - epsilon)) && ((position.y < (bboxMin.y + epsilon) || position.y >            (bboxMax.y - epsilon)) || (position.z < (bboxMin.z + epsilon) || position.z > (bboxMax.z - epsilon))) ||
            (position.y < (bboxMin.y + epsilon) || position.y > (bboxMax.y - epsilon)) && ((position.x < (bboxMin.x + epsilon) || position.x > (bboxMax.x - epsilon)) || (position.z < (bboxMin.z + epsilon) || position.z > (bboxMax.z - epsilon))) ||
            (position.z < (bboxMin.z + epsilon) || position.z > (bboxMax.z - epsilon)) && ((position.x < (bboxMin.x + epsilon) || position.x > (bboxMax.x - epsilon)) || (position.y < (bboxMin.y + epsilon) || position.y > (bboxMax.y - epsilon))))
            
            color = make_float4(1.f, 0.f, 0.f, 1.f);
    }


    //---------------------------------------------------------------------------------------------
    //---------------------------------------------------------------------------------------------

    //---------------------------------------------------------------------------------------------
    void update_CUDA_inv_view_model_matrix(const float *invViewMatrix)
    {
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_invModelViewMatrix, invViewMatrix, 3 * sizeof(float4)));
    }

    //---------------------------------------------------------------------------------------------
    glm::mat3 update_CUDA_inv_view_model_matrix_offline(const glm::mat4 &viewMatrix, const glm::mat4 &volumeMatrix)
    {
        glm::mat4 invModelViewMat = volumeMatrix * viewMatrix;
        // glm::mat4 invModelViewMat = glm::inverse(modelViewMat); // if the model-view matrix (ignoring the translation) is orthonormal (i.e. consists only of rotations), then its inverse is just its transpose, which is much cheaper to compute. https://www.opengl.org/discussion_boards/showthread.php/195803-Getting-World-Space-Eye-Vector

        // send to the GPU the inverse model-vue matrix
        float4 invModelViewMatrix[3];
        invModelViewMatrix[0] = make_float4(invModelViewMat[0].x, invModelViewMat[1].x, invModelViewMat[2].x, invModelViewMat[3].x);
        invModelViewMatrix[1] = make_float4(invModelViewMat[0].y, invModelViewMat[1].y, invModelViewMat[2].y, invModelViewMat[3].y);
        invModelViewMatrix[2] = make_float4(invModelViewMat[0].z, invModelViewMat[1].z, invModelViewMat[2].z, invModelViewMat[3].z);

        CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_invModelViewMatrix, invModelViewMatrix, 3 * sizeof(float4)));

        // return the upper left 3x3 matrix
        return glm::mat3(invModelViewMat);
    }

    cudaArray *d_volumeArray;
    //---------------------------------------------------------------------------------------------
    void create_CUDA_transfer_function(cudaTextureObject_t &tfTexture, cudaArray *d_transferFuncArray)
    {
        // Specify texture
        struct cudaResourceDesc resDescTex;
        memset(&resDescTex, 0, sizeof(resDescTex));
        resDescTex.resType = cudaResourceTypeArray;
        resDescTex.res.array.array = d_transferFuncArray;

        // Specify texture object parameters
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.normalizedCoords = 1;
        texDesc.addressMode[0] = cudaAddressModeClamp;

        CUDA_SAFE_CALL(cudaCreateTextureObject(&tfTexture, &resDescTex, &texDesc, NULL));
    }

    //---------------------------------------------------------------------------------------------
    void update_CUDA_transfer_function(tdns::math::Vector4f *transferFunctionPtr, uint32_t transferFunctionWidth, cudaArray *d_transferFuncArray)
    {
        int32_t pitch = transferFunctionWidth * sizeof(float4);
        CUDA_SAFE_CALL(cudaMemcpy2DToArray(d_transferFuncArray, 0, 0, transferFunctionPtr, pitch, pitch, 1, cudaMemcpyHostToDevice));
    }
    //---------------------------------------------------------------------------------------------
    void destroy_CUDA_transfer_function()
    {
        // CUDA_SAFE_CALL(cudaFreeArray(d_transferFuncArray));
    }
} // namespace graphics
} // namespace tdns
