#pragma once

#include <cstdint>
#include <cuda.h>

#include <GcCore/libMath/Vector.hpp>

namespace tdns
{
namespace common
{
    /**
    * @brief Kernel class to call in a kernel to use a 3D array.
    *        
    * @template Type of data to store.
    */
    template<typename T>
    class K_DynamicArray3dDevice
    {
    public:

        /**
        * @brief Force creation of the default constructor.
        */
        K_DynamicArray3dDevice() = default;
        
        /**
        * @brief Create the object from existing data.
        *
        * @param Pointer to the array.
        * @param 3D size of the array.
        */
        K_DynamicArray3dDevice(T *data, const tdns::math::Vector3ui &size);
 
        /**
        * @brief Destructor.
        */
        ~K_DynamicArray3dDevice() = default;

        /**
        * @brief Device getter on the data array.
        * 
        * @return A pointer on data of the 3D GPU array.
        */
        __device__ T* data();

        /**
         * @brief
         */
        __device__ size_t size();

        /**
        * @brief Device operator overload.
        * 
        * @param  position Position inside the data array.
        * 
        * @return The data at the given position in the array.
        */
        __device__ T& operator [] (const size_t position);
        __device__ const T& operator [] (const size_t position) const;

        __device__ T& operator () (const uint3 &position);
        __device__ const T& operator () (const uint3 &position) const;

        __device__ T& operator () (const size_t x, const size_t y, const size_t z);
        __device__ const T& operator () (const size_t x, const size_t y, const size_t z) const;

    protected:
        /**
        * Member data.
        */
        uint32_t    _xyProduct; ///< The product of the x and y size of the array.
        uint3       _size;      ///< 3D size of the array.
        T           *_data;     ///< Data of the 3D GPU array.
    };

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    K_DynamicArray3dDevice<T>::K_DynamicArray3dDevice(T *data, const tdns::math::Vector3ui &size)
    {
        _xyProduct = size[0] * size[1];

        _size = *reinterpret_cast<const uint3*>(size.data());

        _data = data;
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline __device__
    T* K_DynamicArray3dDevice<T>::data()
    {
        return _data;
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline __device__
     size_t K_DynamicArray3dDevice<T>::size()
     {
         return static_cast<size_t>(_size.x) * static_cast<size_t>(_size.y) * static_cast<size_t>(_size.z);
     }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline __device__
    T& K_DynamicArray3dDevice<T>::operator [] (const size_t position)
    { return _data[position]; }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline __device__
    const T& K_DynamicArray3dDevice<T>::operator [] (const size_t position) const
    { return _data[position]; }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline __device__
    T& K_DynamicArray3dDevice<T>::operator () (const uint3 &position)
    {
        return _data[position.x + position.y * _size.x + position.z * _xyProduct];
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline __device__
    const T& K_DynamicArray3dDevice<T>::operator () (const uint3 &position) const
    {
        return _data[position.x + position.y * _size.x + position.z * _xyProduct];
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline __device__
    T& K_DynamicArray3dDevice<T>::operator () (const size_t x, const size_t y, const size_t z)
    {
        return _data[x + y * _size.x + z * _xyProduct];
    }

    //---------------------------------------------------------------------------------------------------
    template<typename T>
    inline __device__
    const T& K_DynamicArray3dDevice<T>::operator () (const size_t x, const size_t y, const size_t z) const
    {
        return _data[x + y * _size.x + z * _xyProduct];
    }
} // namespace tdns
} // namespace common