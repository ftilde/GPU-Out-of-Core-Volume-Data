#include <GcCore/libData/BricksManager.hpp>

#include <memory>
#include <fstream>
#include <iterator>

#include <lz4hc.h>
#include <lz4.h>

#include <GcCore/libCommon/FileSystem.hpp>
#include <GcCore/libCommon/Logger/Logger.hpp>
#include <GcCore/libCommon/Memory.hpp>

namespace tdns
{
namespace data
{
    //---------------------------------------------------------------------------------------------------
    BricksManager::BricksManager(const std::string &volumeDirectory, const tdns::math::Vector3ui &brickSize,
        const tdns::math::Vector3ui &bigBrickSize, uint32_t numberEncodedBytes, size_t cacheSize /* = 32768 */) :
    _brickEdgeSize(brickSize),
    _bigBrickSize(bigBrickSize),
    _numberEncodedBytes(numberEncodedBytes),
    _volumeDirectory(volumeDirectory),
    _cache(cacheSize)
    {}

    //---------------------------------------------------------------------------------------------------
    Brick* BricksManager::get_brick(uint32_t level, const tdns::math::Vector3ui &position)
    {
        LOGDEBUG(10, tdns::common::log_details::Verbosity::INSANE, "Get brick Level [" << level
            << "] position [" << position[0] << " - " << position[1] << " - " << position[2] << "].");

        Bkey key = get_key(level, position);
        // check in cache
        auto it = _bricks.find(key);
        if (it != _bricks.end())
        {
            _cache.update(key);
            return it->second.get();
        }

        //load the brick from the HDD.
        return load_brick(key, level, position);
    }

    //---------------------------------------------------------------------------------------------------
    BricksManager::BrickStatus BricksManager::get_brick(uint32_t level, const tdns::math::Vector3ui &position, Brick **brick)
    {
        LOGDEBUG(10, tdns::common::log_details::Verbosity::INSANE, "Get brick Level [" << level
        << "] position [" << position[0] << " - " << position[1] << " - " << position[2] << "].");

        Bkey key = get_key(level, position);

        //Search in empty list
        {
            auto it = _emptyBricks.find(key);
            if(it != _emptyBricks.end()) return BrickStatus::Empty;
        }

        //check in cache
        {
            auto it = _bricks.find(key);
            if (it != _bricks.end())
            {
                { // rustine
                    std::lock_guard<std::mutex> guard(_lock);
                    _cache.update(key);
                }
                *brick = it->second.get();
                return BrickStatus::Success;
            }
        }

        //load the brick from the storage device.
        *brick = load_brick(key, level, position);
        return *brick ? BrickStatus::Success : BrickStatus::Unknown;
    }

    //---------------------------------------------------------------------------------------------------
    float BricksManager::write_brick(const std::string &outputDirectory,
        const Brick &brick,
        const tdns::math::Vector3ui &brickSize,
        bool compression /*= true*/)
    {
        std::string filePath = get_brick_path(outputDirectory, brickSize, brick.get_level(), brick.get_position());
        std::ofstream os(filePath, std::ios::out | std::ofstream::binary);

        tdns::math::Vector3ui brickEdgeSize = brick.get_edge_size();
        uint32_t nbBytes = brickEdgeSize[0] * brickEdgeSize[1] * brickEdgeSize[2] * brick.get_encoded();

        float ratio = 0.f;
        if(compression && nbBytes < LZ4_MAX_INPUT_SIZE) // LZ4_MAX_INPUT_SIZE = 2 113 929 216 bytes (almost 2Go) max size for compression 
        {
            // LZ4_compress_default() compress faster when dest buffer size is >= LZ4_compressBound(srcSize)
            uint64_t maxCompressedSize = LZ4_compressBound(nbBytes);
            std::vector<uint8_t> dataCompressed(maxCompressedSize);
            uint64_t compressedSize = LZ4_compress_HC(
                reinterpret_cast<const char*>(brick.get_data().data()),
                reinterpret_cast<char *>(dataCompressed.data()),
                nbBytes,
                static_cast<int>(maxCompressedSize),
                9);

            std::copy(dataCompressed.begin(), dataCompressed.begin() + compressedSize, std::ostreambuf_iterator<char>(os));
            ratio = (100.f - ((compressedSize * 100.f) / nbBytes));
        }
        else
        {
            std::copy(brick.get_data().begin(), brick.get_data().end(), std::ostreambuf_iterator<char>(os));
        }

        return ratio;
    }

    //---------------------------------------------------------------------------------------------------
    void BricksManager::load(const MetaData &metaData)
    {
        _bricks.clear();
        _emptyBricks.clear();
        _cache.clear();

        const std::vector<tdns::data::Bkey> &emptyBricks = metaData.get_empty_bricks();
        for(auto it = emptyBricks.begin(); it != emptyBricks.end(); ++it)
            _emptyBricks.insert(*it);
    }

    //---------------------------------------------------------------------------------------------------
    std::string BricksManager::get_status_string(BricksManager::BrickStatus status) const
    {
        switch(status)
        {
            case BricksManager::BrickStatus::Success:
                return "Success";
            case BricksManager::BrickStatus::Unknown:
                return "Unknown";
            case BricksManager::BrickStatus::Empty:
                return "Empty";
            default:
                return "Error Status";
        }
    }

    //---------------------------------------------------------------------------------------------------
    void BricksManager::check_level_directory(const std::string &volumeDirectory, uint32_t level,
        const tdns::math::Vector3ui &brickSize)
    {
        std::string bricksDirectory = volumeDirectory + get_brick_folder(brickSize);
        if (!tdns::common::is_dir(bricksDirectory)) tdns::common::create_folder(bricksDirectory);

        std::string levelDirectory = bricksDirectory + "/L" + std::to_string(level);
        if (!tdns::common::is_dir(levelDirectory)) tdns::common::create_folder(levelDirectory);
    }

    //---------------------------------------------------------------------------------------------------
    std::string BricksManager::get_brick_folder(const tdns::math::Vector3ui &brickSize)
    {
        return "bricks_" + std::to_string(brickSize[0]) + "_" + std::to_string(brickSize[1]) + "_" + std::to_string(brickSize[2]);
    }
    
    //---------------------------------------------------------------------------------------------------
    void BricksManager::insert_in_cache(const Bkey &key, Brick *brick)
    {
        auto oldestValue = _cache.push_back(key, brick);
        if (!oldestValue) return; //nothing else to do 
        
        auto it = _bricks.find(oldestValue->first);
        if (it != _bricks.end())
        {
            _bricks.erase(it);
        }
    }

    //---------------------------------------------------------------------------------------------------
    std::string BricksManager::get_brick_path(const std::string &baseDirectory,
        const tdns::math::Vector3ui &brickSize,
        uint32_t level,
        const tdns::math::Vector3ui &position)
    {
        return baseDirectory + get_brick_folder(brickSize) + "/"
            + get_level_folder(level) + get_brick_name(level, position) + ".raw";
    }

    //---------------------------------------------------------------------------------------------------
    std::string BricksManager::get_level_folder(uint32_t level)
    {
        return "L" + std::to_string(level) + "/";
    }

    //---------------------------------------------------------------------------------------------------
    Brick* BricksManager::load_brick(Bkey key, uint32_t level, const tdns::math::Vector3ui &position)
    {
        //load the brick
        std::string filePath = get_brick_path(_volumeDirectory, _brickEdgeSize, level, position);
        if (!tdns::common::exists(filePath))
        {
            LOGERROR(10, "Unable to load the asked brick. The file path does not exist. path = [" << filePath << "]");
            return nullptr;
        }

        std::ifstream is(filePath, std::ios::in | std::ifstream::binary);
        std::unique_ptr<Brick> brick = tdns::common::create_unique_ptr<Brick>(_brickEdgeSize * _bigBrickSize, _numberEncodedBytes);
        
        // LOAD COMPRESSED BRICK
        std::vector<uint8_t> compressedData;
        compressedData.assign(std::istreambuf_iterator<char>(is), std::istreambuf_iterator<char>());
        LZ4_decompress_fast(
            reinterpret_cast<char*>(compressedData.data()),
            reinterpret_cast<char*>(brick->get_data().data()),
            static_cast<int>(brick->get_data().size()));

        // LOAD NOT COMPRESSED BRICK
        // brick->get_data().assign(std::istreambuf_iterator<char>(is), std::istreambuf_iterator<char>());
        
        brick->set_level(level);
        brick->set_position(position);

        Brick *ptr = brick.get();
        
        {
            std::lock_guard<std::mutex> guard(_lock);
            _bricks[key].swap(brick);
            insert_in_cache(key, ptr);
        }
        
        return ptr;
    }
} //namespace data
} //namespace tdns