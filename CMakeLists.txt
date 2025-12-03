cmake_minimum_required(VERSION 3.10)
project(ppf)

# 设置C++标准（gtl需要C++17）
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找OpenMP
find_package(OpenMP REQUIRED)

if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# 第三方库路径配置（全部使用本地路径，无网络依赖）
set(EIGEN3_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen)
set(XSIMD_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/xsimd/include)
set(KDTREE_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/kdtree)
set(RPLY_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/rply/rply)
set(RPLY_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/rply/rply/rply.c)

# 手动指定本地 gtl 头文件路径（关键：离线环境核心配置）
set(GTL_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/gtl/include")

# 构建ppf库
add_library(ppf SHARED
    ppf.h ppf.cpp 
    util.h util.cpp 
    type.h type.cpp 
    icp.h icp.cpp 
    privateUtil.h 
    filePLY.h filePLY.cpp 
    ${RPLY_SOURCE} 
    serialize.h serialize.cpp 
    privateType.h
)

# 头文件搜索路径（添加本地 gtl/include 目录）
target_include_directories(ppf PRIVATE
    ${KDTREE_INCLUDE_DIRS}
    ${XSIMD_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${RPLY_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${GTL_INCLUDE_DIR}  # 本地 gtl 头文件路径
)

# 链接依赖库（gtl 是头文件库，无需链接，仅需包含头文件）
target_link_libraries(ppf PUBLIC 
    OpenMP::OpenMP_CXX 
)

# 编译选项
target_compile_options(ppf PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/bigobj /arch:AVX2>
    $<$<CXX_COMPILER_ID:GNU>: -mavx -fPIC -fvisibility=hidden -Wl,--exclude-libs,ALL>
)

# 编译定义
target_compile_definitions(ppf PUBLIC API_EXPORTS)

# 别名和安装配置
add_library(ppf::ppf ALIAS ppf)
set_property(TARGET ppf PROPERTY EXPORT_NAME ppf::ppf)

install(FILES ppf.h type.h apiExport.h filePLY.h util.h TYPE INCLUDE)
install(TARGETS ppf EXPORT ppf LIBRARY)

# 可选：构建示例
option(BUILD_EXAMPLE "build example" ON)

if(BUILD_EXAMPLE)
    add_executable(test_ppf main.cpp)
    target_include_directories(test_ppf PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR} 
        ${EIGEN3_INCLUDE_DIRS}
        ${GTL_INCLUDE_DIR}  # 示例也需要本地 gtl 头文件
    )
    target_link_libraries(test_ppf ppf::ppf)
endif(BUILD_EXAMPLE)