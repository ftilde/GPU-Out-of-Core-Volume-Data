#pragma once

#include <GcCore/libData/MetaData.hpp>
#include <string>

namespace tdns
{
namespace app
{
    /**
    * @brief Application class, that run our application.
    */
    class Application
    {
    public:
        /**
        * @brief Initialize the application.
        *
        * Intialize the application by loading the configuration, adding 
        * categories to the logger and so on.
        *
        * @return True if all is good, false otherwise.
        */
        bool init(const std::string& cfg) const;

        /**
        * @brief Start the application.
        */
        void run(const std::string& cfg);

    protected:

        /**
        * @brief Check if the folder "Data" exist next to the binary.
        * If the folder does not exist it creates it.
        *
        * @return True if all is ok, false otherwise.
        */
        bool data_folder_check() const;

        /**
        * @brief Preprocess a whole volume. It creates the pyramid and
        * bricks it.
        */
        void pre_process(tdns::data::MetaData &volumeData) const;

        void pre_process_mipmapping(tdns::data::MetaData &volumeData) const;

        /**
        */
        void pre_process_bricking(tdns::data::MetaData &volumeData, const std::vector<tdns::math::Vector3ui> &levels) const;
        
    };
} // namespace app
} // namespace tdns
