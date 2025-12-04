# PPF Surface Match Python 使用指南

# PPF Surface Match Python User Guide

本文档介绍如何使用PPF (Point Pair Feature) 表面匹配库的Python接口进行3D物体识别和姿态估计。

This document describes how to use the Python interface of the PPF (Point Pair Feature) surface matching library for 3D object recognition and pose estimation.

## 快速开始

## Quick Start

### 1. 安装和构建

### 1. Installation and Build

#### 方法一：使用安装脚本（推荐）

#### Method 1: Using Installation Script (Recommended)

```bash
cd python
./install.sh
```
```bash 
cd python
pip install -e .
```

#### 方法二：手动构建

#### Method 2: Manual Build

```bash
cd python
pip install -r requirements.txt
mkdir build && cd build
cmake .. -DPYTHON_EXECUTABLE=$(which python)
make -j$(nproc)
cd ..
cp build/ppf.so .
```

### 2. 基本使用

### 2. Basic Usage

```python
from ppf_wrapper import PPFMatcher, PointCloud

# 加载模型和场景
# Load model and scene
model = PointCloud.from_file("model.ply")
scene = PointCloud.from_file("scene.ply")

# 创建匹配器并训练模型
# Create matcher and train model
matcher = PPFMatcher()
matcher.train_model(model, sampling_distance_rel=0.025)

# 执行匹配
# Execute matching
matches = matcher.match_scene(
    scene,
    sampling_distance_rel=0.025,
    key_point_fraction=0.1,
    min_score=0.1,
    num_matches=5
)

# 处理结果
# Process results
for i, (pose, score) in enumerate(matches):
    print(f"匹配 {i+1}: 分数 = {score:.4f}")
    print("姿态矩阵:")
    print(pose)
```

## 详细API说明

## Detailed API Documentation

### PointCloud 类

### PointCloud Class

用于表示3D点云数据。
Used to represent 3D point cloud data.

#### 创建点云

#### Creating Point Cloud

```python
# 从PLY文件加载
# Load from PLY file
pc = PointCloud.from_file("model.ply")

# 从NumPy数组创建
# Create from NumPy array
import numpy as np
points = np.random.rand(1000, 3).astype(np.float32)  # Nx3点坐标 / point coordinates
normals = np.random.rand(1000, 3).astype(np.float32)  # Nx3法向量（可选） / normals (optional)
pc = PointCloud.from_numpy(points, normals)
```

#### 属性和方法

#### Properties and Methods

```python
# 基本属性
# Basic properties
print(f"点数: {pc.num_points}")
print(f"是否有法向量: {pc.has_normals}")

# 设置视点（用于法向量计算）
# Set viewpoint (for normal computation)
pc.set_view_point(x, y, z)

# 保存到文件
# Save to file
pc.save("output.ply")

# 转换为NumPy数组
# Convert to NumPy array
points, normals = pc.to_numpy()
```

### PPFMatcher 类

### PPFMatcher Class

主要的匹配器类，用于训练模型和执行匹配。
The main matcher class for training models and performing matching.

#### 训练模型

#### Training Model

```python
matcher = PPFMatcher()

# 基本训练
# Basic training
matcher.train_model(model, sampling_distance_rel=0.025)

# 自定义训练参数
# Custom training parameters
matcher.train_model(
    model,
    sampling_distance_rel=0.025,           # 采样距离（相对于物体直径） / Sampling distance (relative to object diameter)
    feat_distance_step_rel=0.04,           # 特征距离步长 / Feature distance step
    feat_angle_resolution=30,              # 特征角度分辨率 / Feature angle resolution
    pose_ref_rel_sampling_distance=0.01,  # 姿态精化采样距离 / Pose refinement sampling distance
    knn_normal=10,                         # 法向量估计的最近邻数 / KNN for normal estimation
    smooth_normal=True                     # 是否平滑法向量 / Whether to smooth normals
)
```

#### 执行匹配

#### Performing Matching

```python
# 基本匹配
# Basic matching
matches = matcher.match_scene(scene)

# 自定义匹配参数
# Custom matching parameters
matches = matcher.match_scene(
    scene,
    sampling_distance_rel=0.025,           # 场景采样距离 / Scene sampling distance
    key_point_fraction=0.1,                # 关键点比例 / Key point fraction
    min_score=0.1,                         # 最小分数 / Minimum score
    num_matches=5,                         # 最大匹配数 / Maximum number of matches
    knn_normal=10,                         # 法向量估计的最近邻数 / KNN for normal estimation
    smooth_normal=True,                    # 是否平滑法向量 / Whether to smooth normals
    invert_normal=False,                   # 是否反转法向量 / Whether to invert normals
    max_overlap_dist_rel=0.5,              # 最大重叠距离 / Maximum overlap distance
    sparse_pose_refinement=True,           # 稀疏姿态精化 / Sparse pose refinement
    dense_pose_refinement=True,            # 密集姿态精化 / Dense pose refinement
    pose_ref_num_steps=5,                  # 姿态精化步数 / Pose refinement steps
    pose_ref_dist_threshold_rel=0.1,       # 姿态精化距离阈值 / Pose refinement distance threshold
    pose_ref_scoring_dist_rel=0.01         # 评分距离阈值 / Scoring distance threshold
)
```

#### 模型保存和加载

#### Model Saving and Loading

```python
# 保存训练好的模型
# Save trained model
matcher.save_model("trained_model.ppf")

# 加载预训练模型
# Load pre-trained model
new_matcher = PPFMatcher()
new_matcher.load_model("trained_model.ppf")
```

### 工具函数

### Utility Functions

```python
from ppf_wrapper import transform_pointcloud, sample_mesh, compute_bounding_box

# 变换点云
# Transform point cloud
transformed = transform_pointcloud(pc, pose_matrix, use_normal=True)

# 采样网格
# Sample mesh
sampled = sample_mesh(pc, radius=0.01)

# 计算边界框
# Compute bounding box
min_point, max_point = compute_bounding_box(pc)
```

## 完整示例

## Complete Examples

### 示例1：基本物体识别

### Example 1: Basic Object Recognition

```python
#!/usr/bin/env python3

import numpy as np
from ppf_wrapper import PPFMatcher, PointCloud, transform_pointcloud

def main():
    # 加载数据
    # Load data
    model = PointCloud.from_file("gear.ply")
    scene = PointCloud.from_file("gear_n35.ply")
    
    # 设置视点
    # Set viewpoint
    model.set_view_point(620, 100, 500)
    scene.set_view_point(-200, -50, -500)
    
    # 训练和匹配
    # Train and match
    matcher = PPFMatcher()
    matcher.train_model(model, sampling_distance_rel=0.025)
    
    matches = matcher.match_scene(
        scene,
        sampling_distance_rel=0.025,
        key_point_fraction=0.1,
        min_score=0.1,
        num_matches=5
    )
    
    print(f"找到 {len(matches)} 个匹配:")
    for i, (pose, score) in enumerate(matches):
        print(f"匹配 {i+1}: 分数 = {score:.4f}")
        
        # 保存变换后的模型
        # Save transformed model
        transformed = transform_pointcloud(model, pose)
        transformed.save(f"result_{i+1}.ply")

if __name__ == "__main__":
    main()
```

### 示例2：使用NumPy数组

### Example 2: Using NumPy Arrays

```python
#!/usr/bin/env python3

import numpy as np
from ppf_wrapper import PPFMatcher, PointCloud

def create_sample_cube():
    """创建一个立方体点云"""
    """Create a cube point cloud"""
    points = []
    for x in [0, 1]:
        for y in [0, 1]:
            for z in [0, 1]:
                points.append([x, y, z])
    
    # 添加噪声
    # Add noise
    points = np.array(points, dtype=np.float32)
    points += np.random.normal(0, 0.01, points.shape).astype(np.float32)
    
    return PointCloud.from_numpy(points)

def main():
    # 创建模型和场景
    # Create model and scene
    model = create_sample_cube()
    model.save("cube_model.ply")
    
    # 创建场景（变换后的模型）
    # Create scene (transformed model)
    pose = np.array([
        [1, 0, 0, 2],
        [0, 1, 0, 1], 
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float64)
    
    scene = transform_pointcloud(model, pose)
    scene.save("cube_scene.ply")
    
    # 执行匹配
    # Perform matching
    matcher = PPFMatcher()
    matcher.train_model(model)
    
    matches = matcher.match_scene(scene, num_matches=3)
    
    for i, (found_pose, score) in enumerate(matches):
        print(f"匹配 {i+1}: 分数 = {score:.4f}")
        print("找到的姿态矩阵:")
        print(found_pose)

if __name__ == "__main__":
    main()
```

## 参数调优指南

## Parameter Tuning Guide

### 采样距离 (sampling_distance_rel)

### Sampling Distance

- **值范围 / Value Range**: 0.01 - 0.1
- **默认值 / Default**: 0.04
- **说明 / Description**: 相对于物体直径的采样距离 / Sampling distance relative to object diameter
- **建议 / Recommendations**: 
  - 小物体使用较小值 (0.02-0.03) / Use smaller values for small objects
  - 大物体使用较大值 (0.05-0.08) / Use larger values for large objects
  - 噪声大的数据使用较大值 / Use larger values for noisy data

### 关键点比例 (key_point_fraction)

### Key Point Fraction

- **值范围 / Value Range**: 0.05 - 0.5
- **默认值 / Default**: 0.2
- **说明 / Description**: 场景中用作关键点的点比例 / Fraction of points used as key points in the scene
- **建议 / Recommendations**:
  - 快速匹配使用较小值 (0.05-0.1) / Use smaller values for fast matching
  - 精确匹配使用较大值 (0.2-0.3) / Use larger values for accurate matching

### 最小分数 (min_score)

### Minimum Score

- **值范围 / Value Range**: 0.0 - 1.0
- **默认值 / Default**: 0.2
- **说明 / Description**: 返回姿态的最小分数阈值 / Minimum score threshold for returned poses
- **建议 / Recommendations**:
  - 噪声大的数据降低阈值 (0.1-0.15) / Lower threshold for noisy data
  - 清洁数据可以提高阈值 (0.3-0.4) / Higher threshold for clean data

## 性能优化

## Performance Optimization

### 1. 内存管理

### 1. Memory Management

```python
# 对大点云进行采样
# Sample large point clouds
sampled_model = sample_mesh(model, radius=0.01)
sampled_scene = sample_mesh(scene, radius=0.01)

# 使用采样后的数据进行匹配
# Use sampled data for matching
matcher.train_model(sampled_model)
matches = matcher.match_scene(sampled_scene)
```

### 2. 模型缓存

### 2. Model Caching

```python
# 训练并保存模型
# Train and save model
matcher = PPFMatcher()
matcher.train_model(model)
matcher.save_model("cached_model.ppf")

# 后续使用时直接加载
# Load directly for subsequent use
matcher = PPFMatcher()
matcher.load_model("cached_model.ppf")
```

### 3. 并行处理

### 3. Parallel Processing

PPF库自动使用OpenMP进行并行处理，无需额外配置。
The PPF library automatically uses OpenMP for parallel processing, no additional configuration needed.

## 故障排除

## Troubleshooting

### 常见问题

### Common Issues

1. **导入错误**
   **Import Error**
   ```
   ImportError: No module named 'ppf'
   ```
   **解决 / Solution**: 确保已正确构建并复制了ppf.so文件 / Ensure ppf.so is properly built and copied

2. **文件未找到**
   **File Not Found**
   ```
   FileNotFoundError: Failed to load PLY file
   ```
   **解决 / Solution**: 检查文件路径是否正确，确保文件存在 / Check file path and ensure file exists

3. **内存不足**
   **Insufficient Memory**
   ```
   MemoryError
   ```
   **解决 / Solution**: 增加采样距离或使用采样函数减少点数 / Increase sampling distance or use sampling function to reduce points

4. **匹配结果为空**
   **Empty Matching Results**
   ```
   Found 0 matches
   ```
   **解决 / Solution**: 
   - 降低min_score阈值 / Lower min_score threshold
   - 增加key_point_fraction / Increase key_point_fraction
   - 检查数据质量和法向量 / Check data quality and normals

### 调试技巧

### Debugging Tips

```python
# 启用详细输出
# Enable verbose output
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查点云质量
# Check point cloud quality
print(f"模型点数: {model.num_points}")
print(f"场景点数: {scene.num_points}")
print(f"模型有法向量: {model.has_normals}")
print(f"场景有法向量: {scene.has_normals}")

# 可视化中间结果
# Visualize intermediate results
model.save("debug_model.ply")
scene.save("debug_scene.ply")
```

## 高级用法

## Advanced Usage

### 自定义法向量计算

### Custom Normal Computation

```python
# 如果点云没有法向量，可以手动计算
# If point cloud has no normals, compute manually
if not model.has_normals:
    # 设置合适的视点
    # Set appropriate viewpoint
    model.set_view_point(620, 100, 500)
    # 库会自动计算法向量
    # Library will automatically compute normals
```

### 批量处理

### Batch Processing

```python
import glob

def batch_match(model_path, scene_dir, output_dir):
    model = PointCloud.from_file(model_path)
    matcher = PPFMatcher()
    matcher.train_model(model)
    
    scene_files = glob.glob(f"{scene_dir}/*.ply")
    
    for scene_file in scene_files:
        scene = PointCloud.from_file(scene_file)
        matches = matcher.match_scene(scene, num_matches=3)
        
        # 保存结果
        # Save results
        base_name = os.path.basename(scene_file).split('.')[0]
        for i, (pose, score) in enumerate(matches):
            transformed = transform_pointcloud(model, pose)
            transformed.save(f"{output_dir}/{base_name}_match_{i+1}.ply")
```

## 许可证

## License

本项目遵循与原始PPF Surface Match库相同的许可证。
This project follows the same license as the original PPF Surface Match library.

## 贡献

## Contributing

欢迎提交问题和增强请求！
Welcome to submit issues and enhancement requests!