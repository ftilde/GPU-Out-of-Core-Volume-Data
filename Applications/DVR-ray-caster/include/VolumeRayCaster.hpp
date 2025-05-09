#pragma once

#include <GcCore/libCommon/CppNorm.hpp>
#include <GcCore/cuda/libGPUCache/CacheManager.hpp>
#undef _Float16
#include <GcCore/libData/MetaData.hpp>

#ifndef __builtin_ia32_ldtilecfg 
#define __builtin_ia32_ldtilecfg exit(123);
#endif

#ifndef __builtin_ia32_sttilecfg
#define __builtin_ia32_sttilecfg exit(123);
#endif

namespace tdns
{
namespace gpucache
{
    template<typename T>
    class CacheManager;

} // namespace gpucache
} // namespace tdns

namespace tdns
{
namespace graphics
{
    /**
    * @brief
    */
    void TDNS_API display_volume_raycaster(tdns::gpucache::CacheManager<uchar1> *manager, tdns::data::MetaData &volumeData);

} // namespace graphics
} // namespace tdns
