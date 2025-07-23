#include "Application.hpp"
#include "VolumeRayCaster.hpp"

#include <vector>
#include <thread>

#include <GcCore/libData/Configuration.hpp>
#include <GcCore/libData/VolumeConfiguration.hpp>
#include <GcCore/cuda/libPreprocessor/Mipmapper.hpp>
#include <GcCore/cuda/libPreprocessor/BrickProcessor.hpp>
#include <GcCore/cuda/libPreprocessor/BrickProcessorPredicate.hpp>
#include <GcCore/libPreprocessor/Bricker_v2.hpp>
#include <GcCore/libCommon/FileSystem.hpp>
#include <GcCore/libCommon/Logger/Logger.hpp>
#include <GcCore/libMath/Vector.hpp>
#include <GcCore/cuda/libGPUCache/CacheManager.hpp>

namespace tdns
{
namespace app
{
    //---------------------------------------------------------------------------------------------
    bool Application::init(const std::string& cfg_file) const
    {
        tdns::data::Configuration &conf = tdns::data::Configuration::get_instance();

        // Load configuration file
        conf.load<tdns::data::TDNSConfigurationParser>(cfg_file);

        if (!data_folder_check())
            return false;

        //step doing when opening a new volume
        std::string fileName, workingDirectory;
        conf.get_field("VolumeFile", fileName);
        conf.get_field("WorkingDirectory", workingDirectory);
        std::string volumeDirectory = workingDirectory + tdns::common::get_file_base_name(fileName) + "/";
        conf.add_field("VolumeDirectory", volumeDirectory);

        return true;
    }

    //---------------------------------------------------------------------------------------------
    void Application::run(const std::string& cfg_file)
    {
        tdns::data::Configuration &conf = tdns::data::Configuration::get_instance();
        std::string volumeDirectory;
        conf.get_field("VolumeDirectory", volumeDirectory);

        int32_t gpuID;
        CUDA_SAFE_CALL(cudaGetDevice(&gpuID));

        tdns::data::MetaData volumeData;

        // Get the brick size in the configuration
        uint32_t brickSize;
        conf.get_field("BrickSize", brickSize);

        uint32_t numBytes;
        conf.get_field("NumberEncodedBytes", numBytes);

        // Create or load the multi-resolution bricked representation of the volume to visualize
        std::string bricksDirectory = volumeDirectory + 
            tdns::data::BricksManager::get_brick_folder(tdns::math::Vector3ui(brickSize));
        if (!tdns::common::is_dir(bricksDirectory))
        {
            LOGINFO(10, tdns::common::log_details::Verbosity::INSANE, 
                "Bricks folder does not exist, start preprocessing... [" << bricksDirectory << "].");
            pre_process(volumeData);
        }
        else if (!volumeData.load()) return;

        LOGINFO(10, tdns::common::log_details::Verbosity::INSANE, "Bricks folder found [" << bricksDirectory << "].");

        // Determine the configuration file of the volume(s) to visualize
        std::vector<tdns::data::VolumeConfiguration> volumeConfigurations(1); // only one volume here
        volumeConfigurations[0] = tdns::data::load_volume_configuration(cfg_file);

        const uint64_t vram_size = 8UL << 30;

        // Cache configuration (size)
        // (here we use only one level of pagination)
        std::vector<tdns::math::Vector3ui> blockSize(1, brickSize);

        uint64_t brickSizeInBytes = brickSize*brickSize*brickSize * numBytes;
        uint64_t numCacheEntries = vram_size / brickSizeInBytes;
        uint64_t initialSizeDim = std::cbrt((double)numCacheEntries);
        tdns::math::Vector3ui cacheSize(initialSizeDim, initialSizeDim, initialSizeDim);
        int d = 0;
        while(((uint64_t)cacheSize[0]) * ((uint64_t)cacheSize[1]) * ((uint64_t)cacheSize[2]) < numCacheEntries) {
            cacheSize[d] += 1;
            d = (d+1)%3;
        }
        //std::cout << cacheSize[0] << ", " << cacheSize[1] << ", " << cacheSize[2] << std::endl;

        std::vector<tdns::math::Vector3ui> cacheSizeVec(1, cacheSize);

        tdns::data::CacheConfiguration cacheConfiguration;
        cacheConfiguration.CacheSize = cacheSizeVec;
        cacheConfiguration.BlockSize = blockSize;
        cacheConfiguration.DataCacheFlags = 1;

        // Create the GPU Cache Manager and run raycaster
        switch (numBytes) {
            case 1: {
                std::unique_ptr<tdns::gpucache::CacheManager<uchar1>> cacheManager;
                cacheManager = tdns::common::create_unique_ptr<tdns::gpucache::CacheManager<uchar1>>(volumeConfigurations[0], cacheConfiguration, gpuID);

                tdns::graphics::display_volume_raycaster(cacheManager.get(), volumeData);
                break;
            }
            case 2: {
                std::unique_ptr<tdns::gpucache::CacheManager<ushort1>> cacheManager;
                cacheManager = tdns::common::create_unique_ptr<tdns::gpucache::CacheManager<ushort1>>(volumeConfigurations[0], cacheConfiguration, gpuID);

                tdns::graphics::display_volume_raycaster(cacheManager.get(), volumeData);
                break;
            }
            case 4: {
                std::unique_ptr<tdns::gpucache::CacheManager<float1>> cacheManager;
                cacheManager = tdns::common::create_unique_ptr<tdns::gpucache::CacheManager<float1>>(volumeConfigurations[0], cacheConfiguration, gpuID);

                tdns::graphics::display_volume_raycaster(cacheManager.get(), volumeData);
                break;
            }
            default:
                LOGFATAL(10, "Invalid value for 'NumberEncodedBytes'");
        }
    }

    //---------------------------------------------------------------------------------------------
    bool Application::data_folder_check() const
    {
        if (tdns::common::is_dir("data")) return true;
        
        if (tdns::common::is_file("data"))
        {
            LOGFATAL(10, "Data already exist next to the binary and is not a folder.");
            return false;
        }

        LOGTRACE(10, tdns::common::log_details::Verbosity::INSANE, "Create the folder \"data\" next to the binary.");
        tdns::common::create_folder("data");
        return true;
    }

    //---------------------------------------------------------------------------------------------
    void Application::pre_process(tdns::data::MetaData &volumeData) const
    {
        tdns::data::Configuration &conf = tdns::data::Configuration::get_instance();
        std::cout << "Start pre-processing (see log file) ..." << std::endl;

        uint32_t numBytes;
        conf.get_field("NumberEncodedBytes", numBytes);

        // Mipmapping
        tdns::preprocessor::Mipmapper mipmapper;
        mipmapper.process(volumeData);

        // Bricking
        std::vector<tdns::math::Vector3ui> levels = volumeData.get_initial_levels();
        pre_process_bricking(volumeData, levels);

        // PROCESS EMPTY BRICKS AND VOLUME HISTOGRAM
        uint32_t *d_threshold;
        uint32_t threshold = 0;
        conf.get_field("emptyBrickThreshold", threshold);

        CUDA_SAFE_CALL(cudaMalloc(&d_threshold, sizeof(uint32_t)));
        CUDA_SAFE_CALL(cudaMemcpy(d_threshold, &threshold, sizeof(uint32_t), cudaMemcpyHostToDevice));

        switch (numBytes) {
            case 1: {
                tdns::preprocessor::BrickProcessor<uchar1> brickProcessor(volumeData);
                brickProcessor.process_histo();
                brickProcessor.process_empty<tdns::preprocessor::DefaultBrickProcessorPredicate>(d_threshold);
                break;
            }
            case 2: {
                tdns::preprocessor::BrickProcessor<ushort1> brickProcessor(volumeData);
                brickProcessor.process_histo();
                brickProcessor.process_empty<tdns::preprocessor::BrickProcessor16BitsPredicate>(d_threshold);
                break;
            }
            case 4: {
                tdns::preprocessor::BrickProcessor<float1> brickProcessor(volumeData);
                // Not available for float1 apparently
                //brickProcessor.process_histo();
                brickProcessor.process_empty<tdns::preprocessor::BrickProcessorF32Predicate>(d_threshold);
                break;
            }
            default:
                LOGFATAL(10, "Invalid value for 'NumberEncodedBytes'");
        }

        volumeData.write_bricks_xml();
    }

    //---------------------------------------------------------------------------------------------
    void Application::pre_process_mipmapping(tdns::data::MetaData &volumeData) const
    {
        tdns::preprocessor::MipmappingConfiguration configuration;
        tdns::data::Configuration &conf = tdns::data::Configuration::get_instance();

        //Volumedirectory
        conf.get_field("VolumeDirectory", configuration.volumeDirectory);
        //volume file name
        conf.get_field("VolumeFile", configuration.volumeFileName);
        //Outputdirectory
        conf.get_field("VolumeDirectory", configuration.outputDirectory);
        //Level dimensions
        conf.get_field("size_X", configuration.levelDimension[0]);
        conf.get_field("size_Y", configuration.levelDimension[1]);
        conf.get_field("size_Z", configuration.levelDimension[2]);
        //Down sampling ratios
        conf.get_field("downScale_X", configuration.downScaleRatio[0]);
        conf.get_field("downScale_Y", configuration.downScaleRatio[1]);
        conf.get_field("downScale_Z", configuration.downScaleRatio[2]);
        //brick Size
        uint32_t brickSize;
        conf.get_field("BrickSize", brickSize);
        configuration.brickSize = tdns::math::Vector3ui(brickSize);
        //EncodedByte
        conf.get_field("NumberEncodedBytes", configuration.encodedBytes);
        //Number of channels
        conf.get_field("NumberChannels", configuration.numberChannels);

        tdns::preprocessor::process_mipmapping(configuration);
        tdns::preprocessor::fill_metaData(volumeData.get_initial_levels(), volumeData.nb_levels(), configuration.levelDimension, configuration.downScaleRatio);
    }

    //---------------------------------------------------------------------------------------------
    void Application::pre_process_bricking(tdns::data::MetaData &volumeData, const std::vector<tdns::math::Vector3ui> &levels) const
    {
        tdns::preprocessor::BrickingConfiguration configuration;
        tdns::data::Configuration &conf = tdns::data::Configuration::get_instance();
        //brick Size
        uint32_t brickSize;
        conf.get_field("BrickSize", brickSize);
        configuration.brickSize = tdns::math::Vector3ui(brickSize);
        //EncodedByte
        conf.get_field("NumberEncodedBytes", configuration.encodedBytes);
        //covering
        conf.get_field("VoxelCovering", configuration.covering);
        //Volumedirectory
        conf.get_field("VolumeDirectory", configuration.volumeDirectory);
        //volume file name
        conf.get_field("VolumeFile", configuration.volumeFileName);
        //Outputdirectory
        conf.get_field("VolumeDirectory", configuration.outputDirectory);
        //compression ?
        configuration.compression = true;
        //big brick size
        tdns::math::Vector3ui bigBrickSize;
        conf.get_field("BigBrickSizeX", bigBrickSize[0]);
        conf.get_field("BigBrickSizeY", bigBrickSize[1]);
        conf.get_field("BigBrickSizeZ", bigBrickSize[2]);
        configuration.bigBrickSize = bigBrickSize;

        //fill volumeData.
        tdns::preprocessor::init_meta_data(volumeData, configuration, levels);

        std::vector<std::thread> threads(levels.size());
        for (uint32_t i = 0; i < threads.size(); ++i)
        {
            threads[i] = std::thread([&, i, configuration]() mutable
            {
                configuration.level = i;
                configuration.levelDimensionX = levels[i][0];
                configuration.levelDimensionY = levels[i][1];
                configuration.levelDimensionZ = levels[i][2];

                configuration.startX = configuration.startY = configuration.startZ = 0;
                configuration.endX = configuration.levelDimensionX = levels[i][0];
                configuration.endY = configuration.levelDimensionY = levels[i][1];
                configuration.endZ = configuration.levelDimensionZ = levels[i][2];

                tdns::preprocessor::process_bricking(configuration);
            });
        }

        for (size_t i = 0; i < threads.size(); ++i)
            if (threads[i].joinable())
                threads[i].join();
    }
} // namespace app
} // namespace tdns
