import os
import math
import time
import pickle
import numpy as np
import open3d as o3d
from multiprocessing import Pool
from scipy.spatial import KDTree
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any

# 常量定义
VERSION = 100
MAGIC = 0x7F27F
maxThreads = 8
M_2PI = 2 * math.pi

# 数据结构定义
@dataclass
class BoundingBox:
    min: np.ndarray
    max: np.ndarray

    def diameter(self) -> float:
        return np.linalg.norm(self.max - self.min)
    
    def center(self) -> np.ndarray:
        return (self.min + self.max) / 2

@dataclass
class Feature:
    ref_ind: int
    alpha_angle: float

@dataclass
class TrainParam:
    feat_distance_step_rel: float = 0.025
    feat_angle_resolution: int = 180
    pose_ref_rel_sampling_distance: float = 0.05
    knn_normal: int = 10
    smooth_normal: bool = True

@dataclass
class MatchParam:
    num_angles: int = 55
    num_distances: int = 10
    use_normals: bool = True
    use_symmetry: bool = False
    angle_threshold: float = 0.5
    min_votes: int = 0
    refine: bool = True
    check_resolution: bool = True
    max_iterations: int = 15
    inlier_threshold: float = 0.3
    knn_normal: int = 10
    invert_normal: bool = False
    sparse_pose_refinement: bool = True
    dense_pose_refinement: bool = True
    pose_ref_num_steps: int = 20
    max_overlap_dist_rel: float = 0.1
    pose_ref_dist_threshold_rel: float = 0.05
    pose_ref_scoring_dist_rel: float = 0.05
    num_matches: int = 5

@dataclass
class MatchResult:
    sampled_scene: o3d.geometry.PointCloud
    key_point: o3d.geometry.PointCloud

@dataclass
class ConvergenceCriteria:
    iterations: int = 30
    reject_dist: float = 0.1
    mse_min: float = 1e-4
    mse_max: float = 1e-2
    tolerance: float = 1e-3

@dataclass
class ConvergenceResult:
    pose: np.ndarray = field(default_factory=lambda: np.eye(4))
    converged: bool = False
    mse: float = np.inf
    iterations: int = 0
    inliner: int = 0

@dataclass
class Pose:
    pose: np.ndarray
    num_votes: int

class Timer:
    def __init__(self, name: str):
        self.name = name
        self.start_time = time.time()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        duration_ms = (end_time - self.start_time) * 1000
        print(f"{self.name} cost: {duration_ms:.2f} ms")

class ICP:
    def __init__(self, criteria: ConvergenceCriteria):
        self.criteria = criteria

    def regist(self, src: o3d.geometry.PointCloud, dst: o3d.geometry.PointCloud, 
               init_pose: np.ndarray = np.eye(4)) -> ConvergenceResult:
        result = ConvergenceResult()
        result.pose = init_pose
        
        for i in range(self.criteria.iterations):
            # 点云变换
            src_transformed = src.transform(result.pose)
            
            # 寻找对应点
            kdtree = KDTree(dst.points)
            distances, indices = kdtree.query(src_transformed.points)
            
            # 过滤离群点
            inliers = distances < self.criteria.reject_dist
            if np.sum(inliers) < 3:
                break
                
            # 计算ICP
            reg = o3d.pipelines.registration.registration_icp(
                src_transformed, dst, self.criteria.reject_dist, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-6,
                    relative_rmse=1e-6,
                    max_iteration=1
                )
            )
            
            # 更新姿态
            result.pose = reg.transformation @ result.pose
            result.iterations += 1
            result.mse = reg.inlier_rmse
            result.inliner = np.sum(reg.correspondence_set)
            
            # 检查收敛
            if result.mse < self.criteria.mse_min:
                result.converged = True
                break
                
        return result

class Detector:
    def __init__(self):
        self.impl = None
        self.model = None
        self.hash_table: Dict[int, List[Feature]] = {}

    def train_model(self, model: o3d.geometry.PointCloud, sampling_distance_rel: float = 0.025,
                   param: TrainParam = TrainParam()):
        # 检查输入
        if sampling_distance_rel < 0 or sampling_distance_rel > 1:
            raise ValueError("Invalid sampling distance ratio")
        
        # 计算边界框
        bbox = model.get_axis_aligned_bounding_box()
        model_bbox = BoundingBox(bbox.min_bound, bbox.max_bound)
        model_diameter = model_bbox.diameter()
        
        # 点云采样
        sample_step = model_diameter * sampling_distance_rel
        sampled_model = self.sample_point_cloud(model, sample_step)
        
        # 法向量估计
        if not sampled_model.has_normals():
            sampled_model.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamKNN(param.knn_normal)
            )
        sampled_model.normalize_normals()
        
        # 计算PPF特征并构建哈希表
        self.hash_table = self.build_ppf_hash(sampled_model, model_diameter, param)
        self.model = sampled_model
        self.impl = {
            "sampling_distance_rel": sampling_distance_rel,
            "param": param,
            "sampled_model": sampled_model,
            "model_diameter": model_diameter
        }

    def build_ppf_hash(self, cloud: o3d.geometry.PointCloud, diameter: float, 
                      param: TrainParam) -> Dict[int, List[Feature]]:
        hash_table = {}
        points = np.asarray(cloud.points)
        normals = np.asarray(cloud.normals)
        size = len(points)
        
        distance_step = diameter * param.feat_distance_step_rel
        angle_step = M_2PI / param.feat_angle_resolution
        
        print(f"Building PPF hash for {size} points...")
        total_pairs = size * (size - 1)
        processed = 0
        
        for i in range(size):
            p1 = points[i]
            n1 = normals[i]
            
            for j in range(size):
                if i == j:
                    continue
                    
                p2 = points[j]
                n2 = normals[j]
                
                # 计算PPF特征
                ppf = self.compute_ppf(p1, n1, p2, n2, distance_step, angle_step)
                alpha = self.compute_alpha(p1, n1, p2)
                
                # 哈希表存储
                key = self.hash_ppf(ppf)
                if key not in hash_table:
                    hash_table[key] = []
                hash_table[key].append(Feature(ref_ind=i, alpha_angle=alpha))
                
                processed += 1
                if processed % 100000 == 0:
                    progress = processed / total_pairs * 100
                    print(f"Progress: {progress:.1f}%")
        
        print(f"PPF hash built with {len(hash_table)} entries")
        return hash_table

    def compute_ppf(self, p1: np.ndarray, n1: np.ndarray, p2: np.ndarray, n2: np.ndarray,
                   distance_step: float, angle_step: float) -> Tuple[float, float, float, float]:
        d = p2 - p1
        dist = np.linalg.norm(d)
        d_normalized = d / dist if dist > 1e-6 else d
        
        n1_dot_d = np.dot(n1, d_normalized)
        n2_dot_d = np.dot(n2, d_normalized)
        n1_dot_n2 = np.dot(n1, n2)
        
        # 量化特征
        f1 = math.floor(dist / distance_step)
        f2 = math.floor((n1_dot_d + 1) / 2 * (1 / angle_step))
        f3 = math.floor((n2_dot_d + 1) / 2 * (1 / angle_step))
        f4 = math.floor((n1_dot_n2 + 1) / 2 * (1 / angle_step))
        
        return (f1, f2, f3, f4)

    def compute_alpha(self, p1: np.ndarray, n1: np.ndarray, p2: np.ndarray) -> float:
        # 计算alpha角度
        v = p2 - p1
        v_normalized = v / np.linalg.norm(v) if np.linalg.norm(v) > 1e-6 else v
        
        # 构建局部坐标系
        t = np.cross(n1, v_normalized)
        t = t / np.linalg.norm(t) if np.linalg.norm(t) > 1e-6 else np.array([1, 0, 0])
        b = np.cross(n1, t)
        
        # 计算角度
        alpha = math.atan2(np.dot(v_normalized, t), np.dot(v_normalized, b))
        return alpha

    def hash_ppf(self, ppf: Tuple[float, float, float, float]) -> int:
        # 哈希函数将PPF特征转换为整数键
        f1, f2, f3, f4 = ppf
        return int(f1 * 1000000 + f2 * 1000 + f3 * 10 + f4)

    def sample_point_cloud(self, cloud: o3d.geometry.PointCloud, step: float) -> o3d.geometry.PointCloud:
        # 基于体素的点云采样
        voxel_size = step
        downsampled = cloud.voxel_down_sample(voxel_size)
        return downsampled

    def match_scene(self, scene: o3d.geometry.PointCloud, sampling_distance_rel: float = 0.025,
                   key_point_fraction: float = 0.1, min_score: float = 0.5,
                   param: MatchParam = MatchParam(),
                   match_result: Optional[MatchResult] = None) -> Tuple[List[np.ndarray], List[float]]:
        if not self.impl:
            raise RuntimeError("No trained model available")
            
        # 场景预处理
        scene_bbox = BoundingBox(
            np.asarray(scene.get_min_bound()),
            np.asarray(scene.get_max_bound())
        )
        model_diameter = self.impl["model_diameter"]
        sample_step = model_diameter * sampling_distance_rel
        
        # 场景采样
        sampled_scene = self.sample_point_cloud(scene, sample_step)
        if not sampled_scene.has_normals():
            sampled_scene.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamKNN(param.knn_normal)
            )
        sampled_scene.normalize_normals()
        
        # 关键点采样
        key_sample_step = math.sqrt(1.0 / key_point_fraction) * sample_step
        key_points = self.sample_point_cloud(sampled_scene, key_sample_step)
        
        # 投票机制
        poses, scores = self.voting_based_matching(
            key_points, sampled_scene, model_diameter, param, min_score
        )
        
        # 结果存储
        if match_result:
            match_result.sampled_scene = sampled_scene
            match_result.key_point = key_points
            
        return poses, scores

    def voting_based_matching(self, key_points: o3d.geometry.PointCloud, scene: o3d.geometry.PointCloud,
                             model_diameter: float, param: MatchParam, min_score: float) -> Tuple[List[np.ndarray], List[float]]:
        # 投票匹配实现
        key_points_np = np.asarray(key_points.points)
        key_normals_np = np.asarray(key_points.normals)
        scene_points = np.asarray(scene.points)
        scene_normals = np.asarray(scene.normals)
        
        scene_kdtree = KDTree(scene_points)
        pose_list = []
        
        for i in range(len(key_points_np)):
            p1 = key_points_np[i]
            n1 = key_normals_np[i]
            
            # 半径搜索
            radius = model_diameter
            indices = scene_kdtree.query_ball_point(p1, radius)
            
            if len(indices) < param.min_votes:
                continue
                
            # 计算场景PPF特征并投票
            for j in indices:
                if i == j:
                    continue
                    
                p2 = scene_points[j]
                n2 = scene_normals[j]
                
                # 计算PPF
                distance_step = model_diameter * self.impl["param"].feat_distance_step_rel
                angle_step = M_2PI / self.impl["param"].feat_angle_resolution
                ppf = self.compute_ppf(p1, n1, p2, n2, distance_step, angle_step)
                alpha_scene = self.compute_alpha(p1, n1, p2)
                
                # 哈希查找
                key = self.hash_ppf(ppf)
                if key in self.hash_table:
                    for feature in self.hash_table[key]:
                        # 角度差计算
                        alpha_diff = feature.alpha_angle - alpha_scene
                        alpha_diff = (alpha_diff + M_2PI) % M_2PI
                        
                        # 生成候选姿态
                        if alpha_diff < param.angle_threshold:
                            # 计算变换矩阵
                            pose = self.compute_pose(
                                self.model, feature.ref_ind, 
                                key_points, i, alpha_diff
                            )
                            pose_list.append(Pose(pose=pose, num_votes=1))
        
        # 聚类和筛选
        clustered_poses = self.cluster_poses(pose_list, model_diameter * 0.1)
        refined_poses = []
        refined_scores = []
        
        # ICP精配准
        icp = ICP(ConvergenceCriteria(param.max_iterations, param.inlier_threshold))
        for pose in clustered_poses:
            reg_result = icp.regist(self.model, scene, pose.pose)
            if reg_result.converged:
                refined_poses.append(reg_result.pose)
                refined_scores.append(1.0 - reg_result.mse)
        
        # 按分数排序
        sorted_indices = np.argsort(refined_scores)[::-1]
        return (
            [refined_poses[i] for i in sorted_indices[:param.num_matches]],
            [refined_scores[i] for i in sorted_indices[:param.num_matches]]
        )

    def compute_pose(self, model: o3d.geometry.PointCloud, model_idx: int, 
                    scene: o3d.geometry.PointCloud, scene_idx: int, alpha: float) -> np.ndarray:
        # 计算从模型到场景的变换矩阵
        model_point = np.asarray(model.points)[model_idx]
        model_normal = np.asarray(model.normals)[model_idx]
        scene_point = np.asarray(scene.points)[scene_idx]
        scene_normal = np.asarray(scene.normals)[scene_idx]
        
        # 平移向量
        t = scene_point - model_point
        
        # 旋转矩阵 (基于角度alpha)
        R = np.array([
            [math.cos(alpha), -math.sin(alpha), 0],
            [math.sin(alpha), math.cos(alpha), 0],
            [0, 0, 1]
        ])
        
        # 组合变换矩阵
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = t
        return pose

    def cluster_poses(self, poses: List[Pose], threshold: float) -> List[Pose]:
        # 姿态聚类
        if not poses:
            return []
            
        clusters = []
        for pose in poses:
            matched = False
            for cluster in clusters:
                # 计算姿态距离
                t_diff = np.linalg.norm(cluster[0].pose[:3, 3] - pose.pose[:3, 3])
                if t_diff < threshold:
                    cluster.append(pose)
                    matched = True
                    break
            if not matched:
                clusters.append([pose])
        
        # 计算平均姿态
        avg_poses = []
        for cluster in clusters:
            if len(cluster) < 3:
                continue
                
            # 平均变换矩阵
            avg_pose = np.mean([p.pose for p in cluster], axis=0)
            total_votes = sum(p.num_votes for p in cluster)
            avg_poses.append(Pose(pose=avg_pose, num_votes=total_votes))
        
        # 按投票数排序
        avg_poses.sort(key=lambda x: x.num_votes, reverse=True)
        return avg_poses

    def save(self, filename: str):
        if not self.impl:
            raise RuntimeError("No trained model to save")
            
        data = {
            "magic": MAGIC,
            "version": VERSION,
            "sampling_distance_rel": self.impl["sampling_distance_rel"],
            "param": self.impl["param"],
            "sampled_model": (
                np.asarray(self.impl["sampled_model"].points),
                np.asarray(self.impl["sampled_model"].normals)
            ),
            "hash_table": self.hash_table
        }
        
        with open(filename, "wb") as f:
            pickle.dump(data, f)

    def load(self, filename: str):
        with open(filename, "rb") as f:
            data = pickle.load(f)
            
        if data["magic"] != MAGIC or data["version"] != VERSION:
            raise RuntimeError("Invalid model file")
            
        # 重建点云
        model = o3d.geometry.PointCloud()
        model.points = o3d.utility.Vector3dVector(data["sampled_model"][0])
        model.normals = o3d.utility.Vector3dVector(data["sampled_model"][1])
        
        self.impl = {
            "sampling_distance_rel": data["sampling_distance_rel"],
            "param": data["param"],
            "sampled_model": model,
            "model_diameter": BoundingBox(
                np.min(data["sampled_model"][0], axis=0),
                np.max(data["sampled_model"][0], axis=0)
            ).diameter()
        }
        self.model = model
        self.hash_table = data["hash_table"]

# 主函数示例
def main():
    print("Starting PPF surface matching...")
    
    # 读取模型和场景点云
    print("Loading point clouds...")
    model = o3d.io.read_point_cloud("gear.ply")
    scene = o3d.io.read_point_cloud("gear_n35.ply")
    print(f"Model points: {len(model.points)}")
    print(f"Scene points: {len(scene.points)}")
    
    # 训练模型
    print("Training model...")
    detector = Detector()
    # 使用更大的采样距离来减少点数
    with Timer("Train model"):
        detector.train_model(model, sampling_distance_rel=0.025)  # 进一步增加采样距离
        detector.save("model.ppf")
    print("Model trained and saved.")
    
    # 匹配场景
    print("Loading model for matching...")
    detector2 = Detector()
    detector2.load("model.ppf")
    
    match_result = MatchResult(sampled_scene=o3d.geometry.PointCloud(), key_point=o3d.geometry.PointCloud())
    print("Matching scene...")
    with Timer("Match scene"):
        poses, scores = detector2.match_scene(scene, match_result=match_result)
    
    print(f"Found {len(poses)} poses")
    
    # 保存结果
    for i, (pose, score) in enumerate(zip(poses, scores)):
        print(f"Pose {i} score: {score}")
        print(pose)
        transformed = model.transform(pose)
        o3d.io.write_point_cloud(f"result_{i}.ply", transformed)
    
    o3d.io.write_point_cloud("sampled_scene.ply", match_result.sampled_scene)
    o3d.io.write_point_cloud("key_points.ply", match_result.key_point)
    print("Processing completed!")

if __name__ == "__main__":
    main()