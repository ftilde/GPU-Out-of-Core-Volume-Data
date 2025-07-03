#include <VolumeRayCaster.hpp>

#include <chrono>
#include <vector>
#include <cmath>
#include <memory>

// GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <GcCore/libData/Configuration.hpp>
#include <GcCore/cuda/libCommon/CudaError.hpp>
#include <GcCore/libCommon/CppNorm.hpp>

#include <Shader.hpp>
#include <Display_helper.hpp>
#include <Camera.hpp>
#include <RayCasters_helpers.hpp>
#include <imgui/imgui_impl_sdl_gl3.h>

#include <SDL2/SDLHelper.hpp>


namespace tdns
{
namespace graphics
{

    enum class CompositingMode {
        DVR,
        MOP,
    };

    //---------------------------------------------------------------------------------------------
    template<typename T>
    void display(   SDLGLWindow &window,
                    Shader &shader,
                    tdns::math::Vector2ui &screenSize,
                    tdns::math::Vector3f &bboxmin, tdns::math::Vector3f &bboxmax,
                    float marchingStep,
                    TransferFunction &tf,
                    tdns::data::MetaData &volumeData,
                    tdns::gpucache::CacheManager<T> *manager,
                    uint3 *d_levelsSize,
                    float3 *d_invLevelsSize,
                    float3 *d_LODBrickSize,
                    float *d_LODStepSize,
                    std::vector<float> &histo,
                    CompositingMode compositingMode
                    );

    template<typename T>
    void get_frame( uint32_t *d_pixelBuffer,
                    cudaTextureObject_t &tfTex,
                    const tdns::math::Vector2ui &screenSize,
                    const tdns::math::Vector3f &bboxmin, const tdns::math::Vector3f &bboxmax,
                    float marchingStep,
                    tdns::gpucache::CacheManager<T> *manager,
                    const Camera &camera,
                    uint3 *d_levelsSize,
                    float3 *d_invLevelsSize,
                    float3 *d_LODBrickSize,
                    float *d_LODStepSize,
                    CompositingMode compositingMode
                    );

    //---------------------------------------------------------------------------------------------
    template<typename T>
    void display_volume_raycaster(tdns::gpucache::CacheManager<T> *manager, tdns::data::MetaData &volumeData)
    {
        // Get the needed fiel in the configuration file
        tdns::data::Configuration &conf = tdns::data::Configuration::get_instance();
        tdns::math::Vector2ui screenSize;
        conf.get_field("ScreenWidth", screenSize[0]);
        conf.get_field("ScreenHeight", screenSize[1]);
        std::string volumeDirectory;
        conf.get_field("VolumeDirectory", volumeDirectory);
        uint32_t brickSize;
        conf.get_field("BrickSize", brickSize);
        std::string compositing;
        conf.get_field("Compositing", compositing);
        CompositingMode compositingMode;
        if(compositing == "MOP") {
            compositingMode = CompositingMode::MOP;
        } else if(compositing == "DVR") {
            compositingMode = CompositingMode::DVR;
        } else {
            std::cout << "Invalid compositing mode '" << compositing << "'" << std::endl;
            exit(124);
        }
        
        // Init SDL
        create_sdl_context(SDL_INIT_EVERYTHING);
        SDLGLWindow sdlWindow("3DNS", SDL_WINDOWPOS_UNDEFINED, 100, screenSize[0], screenSize[1], SDL_WINDOW_RESIZABLE);
        
        // Init GLEW
        glewExperimental = GL_TRUE;
        GLenum err = glewInit();
        if (GLEW_OK != err)
            std::cout << "Failed to initialize GLEW : " << glewGetErrorString(err) << std::endl;
        std::cout << "Status: Using GLEW " << glewGetString(GLEW_VERSION) << std::endl;

        glViewport(0, 0, screenSize[0], screenSize[1]);

        // Init ImGui
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO(); (void)io;
        ImGui_ImplSdlGL3_Init(sdlWindow.get_window());
        ImGui::StyleColorsDark();

        // Init GLSL shader
        Shader shader("shaders/shader.vs", "shaders/shader.fs");

        // Load the pre-computed histogram of the volume
        std::vector<float> &histo = volumeData.get_histo();
        size_t histoSize = histo.size();

        // Create and init a transfer function
        TransferFunction tf = TransferFunction(128);

        // Bounding box
        tdns::math::Vector3f bboxmin(-1.f, -1.f, -1.f);
        tdns::math::Vector3f bboxmax(1.f, 1.f, 1.f);
        tdns::math::Vector3f bboxSize(bboxmax[0] - bboxmin[0], bboxmax[1] - bboxmin[1], bboxmax[2] - bboxmin[2]);
        float bboxSizeMin = std::min(bboxSize[0], std::min(bboxSize[1], bboxSize[2]));
        
        // Sampling rate : in order to accurately reconstruct the original signal from the discrete data we need to take at least two samples per voxel.
        float marchingStep = bboxSizeMin / 256.f;

        // Configure parameters with the number of LOD to give to the kernel ray caster
        /* --------------------------------------------------------------------------------------------------------------------- */
        uint32_t nbLevels = volumeData.nb_levels();
        uint3 *levelsSize = reinterpret_cast<uint3*>(volumeData.get_initial_levels().data());
        std::vector<float3> invLevelsSize(nbLevels);
        std::vector<float3> LODBrickSize(nbLevels);
        std::vector<float> LODStepSize(nbLevels);

        for (size_t i = 0; i < nbLevels; ++i)
        {
            invLevelsSize[i] = make_float3( bboxSizeMin / static_cast<float>(levelsSize[i].x),
                                            bboxSizeMin / static_cast<float>(levelsSize[i].y),
                                            bboxSizeMin / static_cast<float>(levelsSize[i].z));

            LODBrickSize[i] = make_float3((bboxSize[0] / levelsSize[i].x) * brickSize, (bboxSize[0] / levelsSize[i].y) * brickSize, (bboxSize[0] / levelsSize[i].z) * brickSize);
            // Determine the sampling step size according to the LOD in order to have 2 samples per voxels.
            LODStepSize[i] = (bboxSizeMin / (std::min(levelsSize[i].x, std::min(levelsSize[i].y, levelsSize[i].z)))) / 2.f;
        }

        uint3 *d_levelsSize;
        CUDA_SAFE_CALL(cudaMalloc(&d_levelsSize,  nbLevels * sizeof(uint3)));
        CUDA_SAFE_CALL(cudaMemcpy(d_levelsSize, levelsSize, nbLevels * sizeof(uint3), cudaMemcpyHostToDevice));
        float3 *d_invLevelsSize;
        CUDA_SAFE_CALL(cudaMalloc(&d_invLevelsSize,  nbLevels * sizeof(float3)));
        CUDA_SAFE_CALL(cudaMemcpy(d_invLevelsSize, invLevelsSize.data(), nbLevels * sizeof(float3), cudaMemcpyHostToDevice));
        float3 *d_LODBrickSize;
        CUDA_SAFE_CALL(cudaMalloc(&d_LODBrickSize,  nbLevels * sizeof(float3)));
        CUDA_SAFE_CALL(cudaMemcpy(d_LODBrickSize, LODBrickSize.data(), nbLevels * sizeof(float3), cudaMemcpyHostToDevice));
        float *d_LODStepSize;
        CUDA_SAFE_CALL(cudaMalloc(&d_LODStepSize,  nbLevels * sizeof(float)));
        CUDA_SAFE_CALL(cudaMemcpy(d_LODStepSize, LODStepSize.data(), nbLevels * sizeof(float), cudaMemcpyHostToDevice));
        /* --------------------------------------------------------------------------------------------------------------------- */

        /*********************** CALL THE DISPLAY FUNCTION ***********************/
        display(sdlWindow, shader, screenSize, bboxmin, bboxmax, marchingStep, tf, volumeData, manager, d_levelsSize, d_invLevelsSize, d_LODBrickSize, d_LODStepSize, histo, compositingMode);
    }

    template void display_volume_raycaster<uchar1>(tdns::gpucache::CacheManager<uchar1> *manager, tdns::data::MetaData &volumeData);
    template void display_volume_raycaster<ushort1>(tdns::gpucache::CacheManager<ushort1> *manager, tdns::data::MetaData &volumeData);
    template void display_volume_raycaster<float1>(tdns::gpucache::CacheManager<float1> *manager, tdns::data::MetaData &volumeData);

    //---------------------------------------------------------------------------------------------
    template<typename T>
    void display(   SDLGLWindow &window,
                    Shader &shader,
                    tdns::math::Vector2ui &screenSize,
                    tdns::math::Vector3f &bboxmin, tdns::math::Vector3f &bboxmax,
                    float marchingStep,
                    TransferFunction &tf,
                    tdns::data::MetaData &volumeData,
                    tdns::gpucache::CacheManager<T> *manager,
                    uint3 *d_levelsSize,
                    float3 *d_invLevelsSize,
                    float3 *d_LODBrickSize,
                    float *d_LODStepSize,
                    std::vector<float> &histo,
                    CompositingMode compositingMode
                    )
    {
        // Init pixel buffer object and CUDA-OpenGL interoperability ressources
        uint32_t pbo, vbo, vao, texture;
        struct cudaGraphicsResource *cuda_pbo_dest_resource;
        init_GL_ressources_raycaster(pbo, texture, &cuda_pbo_dest_resource, screenSize);
        init_mesh_raycaster(&vbo, &vao);
        
        // Add the camera
        Camera camera;

        // Model and View Matrix
        // glm::mat4 modelMatrix = glm::mat4(1.f); // identity matrix
        // glm::mat4 viewMatrix;
        
        // define the background color
        ImVec4 bgColor = ImVec4(0.1f, 0.1f, 0.1f, 1.f);
            
        // Get the size of the histogram of the volume
        size_t histoSize = histo.size();

        // Map the openGL pixel buffer to accessible CUDA space memory
        uint32_t *d_pixelBuffer;
        size_t num_bytes;
        CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &cuda_pbo_dest_resource, 0));
        CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void **)&d_pixelBuffer, &num_bytes, cuda_pbo_dest_resource));

        // CUDA transfer function
        cudaArray *d_transferFuncArray;
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
        CUDA_SAFE_CALL(cudaMallocArray(&d_transferFuncArray, &channelDesc, 128, 1));
        CUDA_SAFE_CALL(cudaMemcpy2DToArray(d_transferFuncArray, 0, 0, tf.get_samples_data(), 128 * sizeof(float4), 128 * sizeof(float4), 1, cudaMemcpyHostToDevice));
        cudaTextureObject_t tfTex;
	    create_CUDA_transfer_function(tfTex, d_transferFuncArray);

        // Enable and configure openGL alpha transparency 
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        
        // model-view matrix
        float3 viewRotation;
        float3 viewTranslation = make_float3(0.0, 0.0, -2.0f);
        float invViewMatrix[12];
        
        bool run = true;

        // main loop
        while (run)
        {    
            // Clear the colorbuffer
	        glClearColor(bgColor.x, bgColor.y, bgColor.z, bgColor.w);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            update_CUDA_transfer_function(reinterpret_cast<tdns::math::Vector4f*>(tf.get_samples_data()), 128, d_transferFuncArray);

            // Activate shader
            shader.use();
            
            updateViewMatrix(invViewMatrix, viewRotation, viewTranslation);
            // viewMatrix = camera.GetViewMatrix();

            // update_CUDA_inv_view_model_matrix(viewMatrix, modelMatrix);
            update_CUDA_inv_view_model_matrix(invViewMatrix);
            get_frame(d_pixelBuffer, tfTex, screenSize, bboxmin, bboxmax, marchingStep, manager, camera, d_levelsSize,d_invLevelsSize, d_LODBrickSize, d_LODStepSize, compositingMode);

            // Bind VAO
            glBindVertexArray(vao);

            // Activate the texture and give the location to the shader
            glActiveTexture(GL_TEXTURE);
            glBindTexture(GL_TEXTURE_2D, texture);
            glUniform1i(glGetUniformLocation(shader.Program, "tex"), 0);

            // Download texture from destination PBO
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, screenSize[0], screenSize[1], GL_RGBA, GL_UNSIGNED_BYTE, NULL);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

            glDrawArrays(GL_TRIANGLES, 0, 6);

#if TDNS_MODE != TDNS_MODE_DEBUG
            // IMGUI crashes in debug mode for some reason
            render_gui(window, bgColor, bboxmin, bboxmax, histo, histoSize, tf);
#endif

            // Swap buffer
            window.display();

            glBindVertexArray(0);
            glBindTexture(GL_TEXTURE_2D, 0);

            // Update the caches manager
            manager->update();

            // Compute cache completude
            std::vector<float> completude;
            manager->completude(completude);

            // Print window title
            std::string title = "3DNS - [Cache used " + std::to_string(completude[0] * 100.f) + "%]";
            window.set_title(title);

            handle_event(window, viewRotation, viewTranslation, marchingStep, run);
        }

        /********  Cleanup ********/

        CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &cuda_pbo_dest_resource, 0));        

        // destroy_CUDA_volume();
        destroy_CUDA_transfer_function();

        glDeleteVertexArrays(1, &vao);
        glDeleteBuffers(1, &vbo);
        glDeleteBuffers(1, &pbo);
        glDeleteTextures(1, &texture);

        CUDA_SAFE_CALL(cudaFree(d_levelsSize));
        CUDA_SAFE_CALL(cudaFree(d_invLevelsSize));
        CUDA_SAFE_CALL(cudaFree(d_LODBrickSize));
        CUDA_SAFE_CALL(cudaFree(d_LODStepSize));

        ImGui_ImplSdlGL3_Shutdown();
        ImGui::DestroyContext();

        quit_sdl();
    }

    //---------------------------------------------------------------------------------------------
    template<typename T>
    void get_frame( uint32_t *d_pixelBuffer,
                    cudaTextureObject_t &tfTex,
                    const tdns::math::Vector2ui &screenSize,
                    const tdns::math::Vector3f &bboxmin, const tdns::math::Vector3f &bboxmax,
                    float marchingStep,
                    tdns::gpucache::CacheManager<T> *manager,
                    const Camera &camera, 
                    uint3 *d_levelsSize,
                    float3 *d_invLevelsSize,
                    float3 *d_LODBrickSize,
                    float *d_LODStepSize,
                    CompositingMode compositingMode
                    )
    {
        dim3 gridDim = dim3((screenSize[0] % 16 != 0) ? (screenSize[0] / 16 + 1) : (screenSize[0] / 16), (screenSize[1] % 16 != 0) ? (screenSize[1] / 16 + 1) : (screenSize[1] / 16));
        dim3 blockDim(16, 16);

        uint32_t renderScreenWidth = screenSize[0];

        // float theta = tan(camera.get_fov() / 2.f * M_PI / 180.0f);

        // cudaProfilerStart();
        switch(compositingMode) {
            case CompositingMode::DVR:
                RayCastDVR<<<gridDim, blockDim>>>(
                    d_pixelBuffer,
                    tfTex,
                    *reinterpret_cast<const uint2*>(screenSize.data()),
                    renderScreenWidth,
                    camera.get_fov(),
                    *reinterpret_cast<const float3*>(bboxmin.data()), *reinterpret_cast<const float3*>(bboxmax.data()),
                    1024, marchingStep, //1000, 0.0015f    sampleMax, stepSize
                    manager->to_kernel_object(),
                    d_invLevelsSize,
                    d_levelsSize,
                    d_LODBrickSize,
                    d_LODStepSize,
                    time(0));
                break;
            case CompositingMode::MOP:
                RayCastMOP<<<gridDim, blockDim>>>(
                    d_pixelBuffer,
                    tfTex,
                    *reinterpret_cast<const uint2*>(screenSize.data()),
                    renderScreenWidth,
                    camera.get_fov(),
                    *reinterpret_cast<const float3*>(bboxmin.data()), *reinterpret_cast<const float3*>(bboxmax.data()),
                    1024, marchingStep, //1000, 0.0015f    sampleMax, stepSize
                    manager->to_kernel_object(),
                    d_invLevelsSize,
                    d_levelsSize,
                    d_LODBrickSize,
                    d_LODStepSize,
                    time(0));
                break;
        }
        // cudaProfilerStop();

#if TDNS_MODE == TDNS_MODE_DEBUG
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
#endif

        CUDA_CHECK_KERNEL_ERROR();
    }
} // namespace graphics
} // namespace tdns
