#pragma once

#include <string>
#include <vector>

#include <GcCore/libMath/Vector.hpp>
#include <GcCore/libCommon/CppNorm.hpp>
#include <GcCore/libData/BrickKey.hpp>

namespace tdns
{
namespace data
{
    struct TDNS_API VolumeConfiguration
    {
        std::string             VolumeDirectory;    ///< Path where the volume is.
        std::string             VolumeFileName;     ///< Name of the volume (with the extension).
        tdns::math::Vector3ui   BrickSize;          ///< Size of a brick.
        tdns::math::Vector3ui   BigBrickSize;       ///< Number of bricks in a big brick.
        tdns::math::Vector3ui   Covering;           ///< Number of overlapping voxels.
        uint32_t                EncodedBytes;       ///< Number of byte a voxel is encoded on.
        uint32_t                Channels;           ///< Number of byte a voxel is encoded on.
        uint32_t                NbLevels;           ///< Number of levels for the volume.

        std::vector<tdns::math::Vector3ui>  InitialVolumeSizes;
        std::vector<tdns::math::Vector3ui>  RealVolumesSizes;
        std::vector<tdns::math::Vector3ui>  NbBricks;
        std::vector<tdns::math::Vector3ui>  NbBigBricks;
        std::vector<Bkey>                   EmptyBricks;
        std::vector<float>                  Histogram;
    };

    /**
    */
    VolumeConfiguration TDNS_API load_volume_configuration(const std::string &configurationFile);

    /**
    */
    void TDNS_API write_volume_configuration(const std::string &file);
} // namespace data
} // namespace tdns