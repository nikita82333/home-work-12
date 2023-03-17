cmake_minimum_required(VERSION 3.9)

set(PATCH_VERSION "1" CACHE INTERNAL "Patch version")
set(PROJECT_VERSION 0.0.${PATCH_VERSION})

project(fashion_mnist VERSION ${PROJECT_VERSION})

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static")

set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

add_subdirectory(eigen3)
find_package(Eigen3 REQUIRED)
set(EIGEN_BUILD_DOC OFF)
set(EIGEN_BUILD_PKGCONFIG OFF)

add_executable(fashion_mnist_cli main.cpp mlp_classifier.cpp helpers.cpp)

target_link_libraries(fashion_mnist_cli Eigen3::Eigen)

add_subdirectory(googletest)
add_executable(tests test_mlp.cpp mlp_classifier.cpp helpers.cpp)
target_link_libraries(tests gtest_main Eigen3::Eigen)

set_target_properties(fashion_mnist_cli tests PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        )

if (MSVC)
    target_compile_options(fashion_mnist_cli PRIVATE
            /W4
            )
    target_compile_options(tests PRIVATE
            /W4
            )
else ()
    target_compile_options(fashion_mnist_cli PRIVATE
            -Wall -Wextra -pedantic
            )
    target_compile_options(tests PRIVATE
            -Wall -Wextra -pedantic
            )
endif()

install(TARGETS fashion_mnist_cli RUNTIME DESTINATION bin)

set(CPACK_GENERATOR DEB)

set(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${PROJECT_VERSION_PATCH}")

set(CPACK_PACKAGE_CONTACT example@example.com)

include(CPack)

enable_testing()
add_test(tests tests)



