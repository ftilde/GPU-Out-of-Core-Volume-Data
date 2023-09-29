/*
 */
#include <GcCore/libData/RAWFile.hpp>

#include <GcCore/libCommon/FileSystem.hpp>
#include <GcCore/libCommon/Logger/Logger.hpp>
#include <GcCore/libCommon/Memory.hpp>

namespace tdns
{
namespace data
{
    //---------------------------------------------------------------------------------------------------
    RAWFile::RAWFile(const std::string &filePath) : AbstractFile(filePath)
    {}

    //---------------------------------------------------------------------------------------------------
    RAWFile::~RAWFile()
    {
        if (_fileStream.is_open())
            close();
    }

    //---------------------------------------------------------------------------------------------------
    void RAWFile::open()
    {
        if (!tdns::common::exists(_filePath))
            tdns::common::create_file(_filePath);

        _fileStream.open(_filePath, std::fstream::in | std::fstream::out | std::fstream::binary);
    }

    //---------------------------------------------------------------------------------------------------
    void RAWFile::close()
    {
        _fileStream.close();
    }

    //---------------------------------------------------------------------------------------------------
    bool RAWFile::read(uint8_t *buffer, uint32_t size)
    {
        if (!_fileStream.is_open())
        {
            LOGERROR(10, "Cannot read the file ! File is not opened.");
            return false;
        }

        _fileStream.read(reinterpret_cast<char*>(buffer), size);

        return true;
    }

    //---------------------------------------------------------------------------------------------------
    bool RAWFile::write(uint8_t *data, uint32_t size)
    {
        if (!_fileStream.is_open())
        {
            LOGERROR(10, "Cannot write in the file ! File is not opened.");
            return false;
        }

        _fileStream.write(reinterpret_cast<char*>(data), size);

        return true;
    }

    //---------------------------------------------------------------------------------------------------
    bool RAWFile::set_relative_cursor_position(uint64_t offset)
    {
        if (!_fileStream.is_open())
        {
            LOGERROR(10, "Cannot set the cursor position in the file ! File is not opened.");
            return false;
        }

        _fileStream.seekg(offset, std::ios_base::cur);

        return true;
    }

    //---------------------------------------------------------------------------------------------------
    bool RAWFile::set_absolute_cursor_position(uint64_t position)
    {
        if (!_fileStream.is_open())
        {
            LOGERROR(10, "Cannot set the cursor position in the file ! File is not opened.");
            return false;
        }

        _fileStream.seekg(position);

        return true;
    }

    //---------------------------------------------------------------------------------------------------
    std::unique_ptr<RAWFile> RAWFile::create_instance(const std::string &filePath)
    {
        return tdns::common::create_unique_ptr<RAWFile>(filePath);
    }

} //namespace data
} //namespace tdns
