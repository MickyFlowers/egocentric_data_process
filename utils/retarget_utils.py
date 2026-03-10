from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from manopth.manolayer import ManoLayer
from scipy.spatial.transform import Rotation as Rscipy


def load_pose_archive(path):
    archive_path = Path(path).expanduser().resolve()
    return torch.load(str(archive_path), map_location="cpu", weights_only=False)


def to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def ensure_float32(value):
    return to_numpy(value).astype(np.float32, copy=False)


def ensure_bool(value):
    return to_numpy(value).astype(np.bool_, copy=False)


def resolve_torch_device(device_name="cpu"):
    if device_name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def to_torch(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, np.ndarray):
        array = value.astype(np.float32, copy=False) if value.dtype == np.float64 else value
        return torch.from_numpy(array).to(device)
    if isinstance(value, dict):
        return {key: to_torch(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return torch.as_tensor(value, device=device)
    if isinstance(value, (int, float, bool)):
        return torch.as_tensor(value, device=device)
    return value


def resolve_mano_model_dir(mano_model_dir):
    mano_path = Path(mano_model_dir).expanduser().resolve()
    candidates = []
    if mano_path.name == "model":
        candidates.append(mano_path.with_name("models"))
    candidates.append(mano_path / "models")
    if mano_path.parent != mano_path:
        candidates.append(mano_path.parent / "models")
    candidates.append(mano_path)

    for candidate in candidates:
        left_model = candidate / "MANO_LEFT.pkl"
        right_model = candidate / "MANO_RIGHT.pkl"
        if left_model.exists() and right_model.exists():
            return candidate.resolve()

    raise FileNotFoundError(f"MANO model directory not found under: {mano_path}")


@lru_cache(maxsize=8)
def load_mano_layer(mano_model_dir, side, use_pca, flat_hand_mean, fix_shapedirs, device_name):
    mano_path = resolve_mano_model_dir(mano_model_dir)
    layer = ManoLayer(
        mano_root=str(mano_path),
        use_pca=use_pca,
        flat_hand_mean=flat_hand_mean,
        side=side,
    )
    if side == "left" and fix_shapedirs and hasattr(layer, "th_shapedirs"):
        with torch.no_grad():
            layer.th_shapedirs[:, 0, :] *= -1
    layer = layer.to(resolve_torch_device(device_name))
    layer.eval()
    return layer


def mano_forward(mano_layer, global_orient, hand_pose, betas, transl):
    pose_coeffs = torch.cat([global_orient, hand_pose], dim=1)
    with torch.no_grad():
        vertices, joints = mano_layer(
            pose_coeffs.float(),
            th_betas=betas.float(),
            th_trans=transl.float(),
        )
    return vertices / 1000.0, joints / 1000.0


def scale_traj_translation(traj, scale):
    traj_scaled = traj.clone()
    traj_scaled[:, :3] = traj_scaled[:, :3] * scale
    return traj_scaled


def quaternion_xyzw_to_matrix_torch(quaternion):
    quat = quaternion.float()
    quat = quat / torch.clamp(torch.linalg.norm(quat, dim=-1, keepdim=True), min=1e-8)
    qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.stack(
        [
            1 - 2 * (qy**2 + qz**2),
            2 * (qx * qy - qz * qw),
            2 * (qx * qz + qy * qw),
            2 * (qx * qy + qz * qw),
            1 - 2 * (qx**2 + qz**2),
            2 * (qy * qz - qx * qw),
            2 * (qx * qz - qy * qw),
            2 * (qy * qz + qx * qw),
            1 - 2 * (qx**2 + qy**2),
        ],
        dim=1,
    ).view(-1, 3, 3)


def world_to_camera(points_world, traj):
    translation = traj[:, :3]
    rotation = quaternion_xyzw_to_matrix_torch(traj[:, 3:])
    return torch.bmm(points_world - translation.unsqueeze(1), rotation)


def rotation_matrix_to_axis_angle(rotation):
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(matrix).all():
        return np.full((3,), np.nan, dtype=np.float32)

    # Project to the nearest valid rotation matrix to avoid numerical drift.
    u, _, vt = np.linalg.svd(matrix)
    ortho = u @ vt
    if np.linalg.det(ortho) < 0.0:
        u[:, -1] *= -1.0
        ortho = u @ vt
    return Rscipy.from_matrix(ortho).as_rotvec().astype(np.float32, copy=False)


def axis_angle_to_rotation_matrix(axis_angle):
    rotvec = ensure_float32(axis_angle).reshape(3)
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-8:
        return np.eye(3, dtype=np.float32)

    axis = rotvec / angle
    x, y, z = axis.astype(np.float32, copy=False)
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    one_minus_cosine = 1.0 - cosine
    return np.array(
        [
            [cosine + x * x * one_minus_cosine, x * y * one_minus_cosine - z * sine, x * z * one_minus_cosine + y * sine],
            [y * x * one_minus_cosine + z * sine, cosine + y * y * one_minus_cosine, y * z * one_minus_cosine - x * sine],
            [z * x * one_minus_cosine - y * sine, z * y * one_minus_cosine + x * sine, cosine + z * z * one_minus_cosine],
        ],
        dtype=np.float32,
    )


def _rotvec_equivalent_candidates(rotvec, wrap_count=1):
    vector = np.asarray(rotvec, dtype=np.float64).reshape(3)
    if not np.isfinite(vector).all():
        return [vector]
    angle = float(np.linalg.norm(vector))
    if angle < 1e-10:
        return [np.zeros((3,), dtype=np.float64)]

    axis = vector / angle
    period = 2.0 * np.pi
    candidates = []
    for k in range(-int(wrap_count), int(wrap_count) + 1):
        candidates.append(axis * (angle + k * period))
        flipped_angle = (period - angle) + k * period
        candidates.append(-axis * flipped_angle)
    return candidates


def _closest_equivalent_rotvec(rotvec, reference, wrap_count=1):
    target = np.asarray(rotvec, dtype=np.float64).reshape(3)
    ref = np.asarray(reference, dtype=np.float64).reshape(3)
    if not np.isfinite(target).all() or not np.isfinite(ref).all():
        return target
    candidates = _rotvec_equivalent_candidates(target, wrap_count=wrap_count)
    best = target
    best_distance = float(np.linalg.norm(target - ref))
    for candidate in candidates:
        distance = float(np.linalg.norm(candidate - ref))
        if distance < best_distance:
            best = candidate
            best_distance = distance
    return best


KF_EDGE_PAD = 15

# Keep pinch normalization aligned with the reference implementation:
# /Users/cyxovo/Downloads/ego_data_process-main/src/retarget.py
PINCH_DIST_CLOSED_M = 0.05
PINCH_DIST_OPEN_M = 0.10
PINCH_GAMMA = 0.5


def _rotation_x(angle_rad):
    cosine = float(np.cos(angle_rad))
    sine = float(np.sin(angle_rad))
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cosine, -sine],
            [0.0, sine, cosine],
        ],
        dtype=np.float32,
    )


def _rotation_z(angle_rad):
    cosine = float(np.cos(angle_rad))
    sine = float(np.sin(angle_rad))
    return np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _euler_zxz_degrees_to_matrix(angles_deg):
    alpha, beta, gamma = np.deg2rad(ensure_float32(angles_deg).reshape(3))
    return (_rotation_z(alpha) @ _rotation_x(beta) @ _rotation_z(gamma)).astype(np.float32)


def get_default_camera_matrix(camera_elevation_deg=45.0):
    rotation = _euler_zxz_degrees_to_matrix(
        [-90.0, -90.0 - float(camera_elevation_deg), 0.0]
    )
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation
    return transform


def pose_vector_to_matrix(vector):
    pose = ensure_float32(vector).reshape(-1)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = axis_angle_to_rotation_matrix(pose[3:6])
    transform[:3, 3] = pose[:3]
    return transform


def pose_matrix_to_vector(matrix, pinch=np.nan):
    transform = ensure_float32(matrix).reshape(4, 4)
    pose = np.full((7,), np.nan, dtype=np.float32)
    pose[:3] = transform[:3, 3]
    pose[3:6] = rotation_matrix_to_axis_angle(transform[:3, :3])
    pose[6] = float(pinch)
    return pose


def build_pose_matrices(pose_vectors, camera_matrix):
    poses = ensure_float32(pose_vectors)
    camera = ensure_float32(camera_matrix).reshape(4, 4)
    pose_matrices = np.full((poses.shape[0], 4, 4), np.nan, dtype=np.float32)
    valid_mask = np.isfinite(poses[:, :6]).all(axis=1)
    for frame_index in np.flatnonzero(valid_mask):
        pose_matrices[frame_index] = camera @ pose_vector_to_matrix(poses[frame_index])
    return pose_matrices


def compute_pose_bbox(pose_matrices):
    matrices = ensure_float32(pose_matrices)
    valid_mask = np.isfinite(matrices[:, :3, 3]).all(axis=1)
    if not valid_mask.any():
        return None, None
    translations = matrices[valid_mask, :3, 3]
    return translations.min(axis=0), translations.max(axis=0)


def align_poses_to_workstation(
    left_pose_matrices,
    right_pose_matrices,
    workstation_center,
    camera_matrix,
):
    left_matrices = ensure_float32(left_pose_matrices).copy()
    right_matrices = ensure_float32(right_pose_matrices).copy()
    camera = ensure_float32(camera_matrix).reshape(4, 4).copy()
    workstation = ensure_float32(workstation_center).reshape(3)

    combined = []
    left_valid = np.isfinite(left_matrices[:, :3, 3]).all(axis=1)
    right_valid = np.isfinite(right_matrices[:, :3, 3]).all(axis=1)
    if left_valid.any():
        combined.append(left_matrices[left_valid])
    if right_valid.any():
        combined.append(right_matrices[right_valid])
    if not combined:
        return left_matrices, right_matrices, camera

    bbox_min, bbox_max = compute_pose_bbox(np.concatenate(combined, axis=0))
    if bbox_min is None:
        return left_matrices, right_matrices, camera

    center = (bbox_min + bbox_max) * 0.5
    shift = workstation - center
    shift[1] = 0.0

    camera[:3, 3] += shift
    left_matrices[left_valid, :3, 3] += shift
    right_matrices[right_valid, :3, 3] += shift
    return left_matrices, right_matrices, camera


class PoseExtendedKalmanFilter:
    def __init__(
        self,
        dt=1 / 60,
        q_pos=100.0,
        q_rot=100.0,
        r_pos=5e-4,
        r_rot=1e-3,
        innovation_gate_pos=None,
        innovation_gate_rot=1.2,
        outlier_noise_scale=1.0e6,
        rotvec_wrap_count=2,
    ):
        self.dt = float(dt)
        self.sd = 12
        self.od = 6
        self.H = np.zeros((6, 12), dtype=np.float64)
        self.H[:6, :6] = np.eye(6, dtype=np.float64)
        self.innovation_gate_pos = (
            None if innovation_gate_pos is None else float(max(innovation_gate_pos, 0.0))
        )
        self.innovation_gate_rot = (
            None if innovation_gate_rot is None else float(max(innovation_gate_rot, 0.0))
        )
        self.outlier_noise_scale = float(max(outlier_noise_scale, 1.0))
        self.rotvec_wrap_count = int(max(rotvec_wrap_count, 0))

        dt2 = self.dt * self.dt
        self.Q = np.zeros((12, 12), dtype=np.float64)
        for index in range(3):
            self.Q[index, index] = dt2 * dt2 / 4.0 * q_pos
            self.Q[index, index + 6] = dt2 * self.dt / 2.0 * q_pos
            self.Q[index + 6, index] = dt2 * self.dt / 2.0 * q_pos
            self.Q[index + 6, index + 6] = dt2 * q_pos

            rot_index = index + 3
            vel_index = index + 9
            self.Q[rot_index, rot_index] = dt2 * dt2 / 4.0 * q_rot
            self.Q[rot_index, vel_index] = dt2 * self.dt / 2.0 * q_rot
            self.Q[vel_index, rot_index] = dt2 * self.dt / 2.0 * q_rot
            self.Q[vel_index, vel_index] = dt2 * q_rot

        self.R = np.diag([r_pos] * 3 + [r_rot] * 3).astype(np.float64)

    @staticmethod
    def mat_to_pose6d(matrix):
        transform = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
        if not np.isfinite(transform[:3, :4]).all():
            return np.full((6,), np.nan, dtype=np.float64)
        pose = np.zeros((6,), dtype=np.float64)
        pose[:3] = transform[:3, 3]
        pose[3:6] = rotation_matrix_to_axis_angle(transform[:3, :3]).astype(np.float64)
        return pose

    @staticmethod
    def pose6d_to_mat(pose):
        vector = np.asarray(pose, dtype=np.float64).reshape(6)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, 3] = vector[:3]
        transform[:3, :3] = axis_angle_to_rotation_matrix(vector[3:6]).astype(np.float64)
        return transform

    def _unwrap_rotvec(self, observations):
        obs = np.asarray(observations, dtype=np.float64).copy()
        last_valid = None
        for frame_index in range(len(obs)):
            current = obs[frame_index, 3:6]
            if not np.isfinite(current).all():
                continue
            if last_valid is None:
                last_valid = current.copy()
                continue
            best = _closest_equivalent_rotvec(
                current,
                last_valid,
                wrap_count=self.rotvec_wrap_count,
            )
            obs[frame_index, 3:6] = best
            last_valid = best.copy()
        return obs

    def _predict(self, state):
        predicted = np.zeros((12,), dtype=np.float64)
        predicted[:3] = state[:3] + state[6:9] * self.dt
        predicted[3:6] = state[3:6] + state[9:12] * self.dt
        predicted[6:9] = state[6:9]
        predicted[9:12] = state[9:12]
        return predicted

    def _jacobian(self, _state):
        jacobian = np.eye(12, dtype=np.float64)
        jacobian[:3, 6:9] = self.dt * np.eye(3, dtype=np.float64)
        jacobian[3:6, 9:12] = self.dt * np.eye(3, dtype=np.float64)
        return jacobian

    def _innovation(self, measurement, predicted):
        return measurement - self.H @ predicted

    @staticmethod
    def _stabilize_covariance(covariance, max_abs_value=1.0e6):
        cov = np.asarray(covariance, dtype=np.float64)
        cov = 0.5 * (cov + cov.T)
        cov = np.nan_to_num(cov, nan=max_abs_value, posinf=max_abs_value, neginf=-max_abs_value)
        return np.clip(cov, -max_abs_value, max_abs_value)

    def filter_and_smooth(self, pose_matrices, edge_pad=0):
        matrices = np.asarray(pose_matrices, dtype=np.float64)
        if matrices.ndim != 3 or matrices.shape[1:] != (4, 4):
            raise ValueError(f"pose_matrices must be shaped as [N, 4, 4], got {matrices.shape}")
        valid_input_mask = np.isfinite(matrices[:, :3, :4]).all(axis=(1, 2))
        if not valid_input_mask.any():
            return matrices.astype(np.float32, copy=False)

        original_length = matrices.shape[0]
        if edge_pad > 0 and original_length > 1:
            first_valid_index = int(np.flatnonzero(valid_input_mask)[0])
            last_valid_index = int(np.flatnonzero(valid_input_mask)[-1])
            start_reference = matrices[first_valid_index : first_valid_index + 1]
            end_reference = matrices[last_valid_index : last_valid_index + 1]
            start_pad = np.tile(start_reference, (edge_pad, 1, 1))
            end_pad = np.tile(end_reference, (edge_pad, 1, 1))
            matrices = np.concatenate([start_pad, matrices, end_pad], axis=0)

        frame_count = matrices.shape[0]
        observations = np.zeros((frame_count, 6), dtype=np.float64)
        for frame_index in range(frame_count):
            observations[frame_index] = self.mat_to_pose6d(matrices[frame_index])
        observations = self._unwrap_rotvec(observations)
        measurement_valid = np.isfinite(observations).all(axis=1)
        if not measurement_valid.any():
            return matrices.astype(np.float32, copy=False)

        predicted_states = np.zeros((frame_count, 12), dtype=np.float64)
        predicted_covariances = np.zeros((frame_count, 12, 12), dtype=np.float64)
        filtered_states = np.zeros((frame_count, 12), dtype=np.float64)
        filtered_covariances = np.zeros((frame_count, 12, 12), dtype=np.float64)
        jacobians = np.zeros((frame_count, 12, 12), dtype=np.float64)

        state = np.zeros((12,), dtype=np.float64)
        first_valid = int(np.flatnonzero(measurement_valid)[0])
        state[:6] = observations[first_valid]
        if first_valid == 0:
            next_valid_candidates = np.flatnonzero(measurement_valid[1:])
            if next_valid_candidates.size > 0:
                second_valid = int(next_valid_candidates[0] + 1)
                dt_between = float(max((second_valid - first_valid) * self.dt, 1e-8))
                state[6:9] = (observations[second_valid, :3] - observations[first_valid, :3]) / dt_between
                state[9:12] = (observations[second_valid, 3:6] - observations[first_valid, 3:6]) / dt_between
        covariance = np.eye(12, dtype=np.float64)
        covariance[:6, :6] *= 0.01
        covariance[6:, 6:] *= 1.0
        identity = np.eye(12, dtype=np.float64)

        for frame_index in range(frame_count):
            jacobian = self._jacobian(state)
            predicted = self._predict(state)
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                predicted_covariance_raw = jacobian @ covariance @ jacobian.T + self.Q
            predicted_covariance = self._stabilize_covariance(predicted_covariance_raw)

            jacobians[frame_index] = jacobian
            predicted_states[frame_index] = predicted
            predicted_covariances[frame_index] = predicted_covariance

            if measurement_valid[frame_index]:
                measurement = observations[frame_index].copy()
                measurement[3:6] = _closest_equivalent_rotvec(
                    measurement[3:6],
                    predicted[3:6],
                    wrap_count=self.rotvec_wrap_count,
                )
                innovation = self._innovation(measurement, predicted)
                measurement_noise = self.R

                pos_outlier = False
                if self.innovation_gate_pos is not None and self.innovation_gate_pos > 0.0:
                    pos_outlier = float(np.linalg.norm(innovation[:3])) > self.innovation_gate_pos
                rot_outlier = False
                if self.innovation_gate_rot is not None and self.innovation_gate_rot > 0.0:
                    rot_outlier = float(np.linalg.norm(innovation[3:6])) > self.innovation_gate_rot

                if pos_outlier and rot_outlier:
                    state = predicted
                    covariance = predicted_covariance
                    filtered_states[frame_index] = state.copy()
                    filtered_covariances[frame_index] = covariance.copy()
                    continue

                if pos_outlier or rot_outlier:
                    measurement_noise = self.R.copy()
                    if pos_outlier:
                        measurement_noise[:3, :3] *= self.outlier_noise_scale
                    if rot_outlier:
                        measurement_noise[3:6, 3:6] *= self.outlier_noise_scale

                innovation_covariance = self.H @ predicted_covariance @ self.H.T + measurement_noise
                try:
                    kalman_gain = np.linalg.solve(
                        innovation_covariance.T,
                        (predicted_covariance @ self.H.T).T,
                    ).T
                except np.linalg.LinAlgError:
                    kalman_gain = (predicted_covariance @ self.H.T) @ np.linalg.pinv(innovation_covariance)
                kalman_gain = np.nan_to_num(kalman_gain, nan=0.0, posinf=0.0, neginf=0.0)
                state = predicted + kalman_gain @ innovation
                with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                    correction = identity - kalman_gain @ self.H
                    covariance_raw = (
                        correction @ predicted_covariance @ correction.T
                        + kalman_gain @ measurement_noise @ kalman_gain.T
                    )
                covariance = covariance_raw
                covariance = self._stabilize_covariance(covariance)
            else:
                state = predicted
                covariance = predicted_covariance
            filtered_states[frame_index] = state.copy()
            filtered_covariances[frame_index] = covariance.copy()

        smoothed_states = filtered_states.copy()
        smoothed_covariances = filtered_covariances.copy()
        for frame_index in range(frame_count - 2, -1, -1):
            next_predicted_covariance = predicted_covariances[frame_index + 1]
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                gain = (
                    filtered_covariances[frame_index]
                    @ jacobians[frame_index + 1].T
                    @ np.linalg.pinv(next_predicted_covariance)
                )
            if not np.isfinite(gain).all():
                gain = np.nan_to_num(gain, nan=0.0, posinf=0.0, neginf=0.0)
            smoothed_states[frame_index] = filtered_states[frame_index] + gain @ (
                smoothed_states[frame_index + 1] - predicted_states[frame_index + 1]
            )
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                smoothed_covariance_raw = filtered_covariances[frame_index] + gain @ (
                    smoothed_covariances[frame_index + 1] - predicted_covariances[frame_index + 1]
                ) @ gain.T
            smoothed_covariances[frame_index] = smoothed_covariance_raw
            smoothed_covariances[frame_index] = self._stabilize_covariance(smoothed_covariances[frame_index])

        result = np.zeros((frame_count, 4, 4), dtype=np.float32)
        for frame_index in range(frame_count):
            result[frame_index] = self.pose6d_to_mat(smoothed_states[frame_index, :6]).astype(np.float32)

        if edge_pad > 0 and original_length > 1:
            result = result[edge_pad : edge_pad + original_length]
        return result


def smooth_pose_matrices(
    pose_matrices,
    dt,
    edge_pad=KF_EDGE_PAD,
    q_pos=120.0,
    q_rot=120.0,
    r_pos=2e-3,
    r_rot=4e-3,
    innovation_gate_pos=None,
    innovation_gate_rot=1.2,
    outlier_noise_scale=1.0e6,
    rotvec_wrap_count=2,
):
    matrices = ensure_float32(pose_matrices).copy()
    valid_mask = np.isfinite(matrices[:, :3, :4]).all(axis=(1, 2))
    if valid_mask.sum() == 0:
        return matrices

    ekf = PoseExtendedKalmanFilter(
        dt=dt,
        q_pos=q_pos,
        q_rot=q_rot,
        r_pos=r_pos,
        r_rot=r_rot,
        innovation_gate_pos=innovation_gate_pos,
        innovation_gate_rot=innovation_gate_rot,
        outlier_noise_scale=outlier_noise_scale,
        rotvec_wrap_count=rotvec_wrap_count,
    )
    return ekf.filter_and_smooth(matrices, edge_pad=edge_pad)


def transform_pose_matrices(pose_matrices, transform_matrix):
    matrices = ensure_float32(pose_matrices).copy()
    transform = ensure_float32(transform_matrix).reshape(4, 4)
    valid_mask = np.isfinite(matrices[:, :3, 3]).all(axis=1)
    for frame_index in np.flatnonzero(valid_mask):
        matrices[frame_index] = transform @ matrices[frame_index]
    return matrices


def build_transform_matrix(translation=(0.0, 0.0, 0.0), rotation=None):
    transform = np.eye(4, dtype=np.float32)
    transform[:3, 3] = ensure_float32(translation).reshape(3)
    if rotation is not None:
        transform[:3, :3] = ensure_float32(rotation).reshape(3, 3)
    return transform


def express_poses_in_frame(pose_matrices, frame_transform):
    frame = ensure_float32(frame_transform).reshape(4, 4)
    return transform_pose_matrices(pose_matrices, np.linalg.inv(frame).astype(np.float32))


def pose_matrices_to_vectors(pose_matrices, pinch=None):
    matrices = ensure_float32(pose_matrices)
    if pinch is None:
        pinch_values = np.full((matrices.shape[0],), np.nan, dtype=np.float32)
    else:
        pinch_values = ensure_float32(pinch).reshape(-1)

    pose_vectors = np.full((matrices.shape[0], 7), np.nan, dtype=np.float32)
    valid_mask = np.isfinite(matrices[:, :3, 3]).all(axis=1)
    previous_rotvec = None
    for frame_index in np.flatnonzero(valid_mask):
        pose = pose_matrix_to_vector(
            matrices[frame_index],
            pinch=pinch_values[frame_index] if frame_index < len(pinch_values) else np.nan,
        )
        if previous_rotvec is not None and np.isfinite(pose[3:6]).all():
            pose[3:6] = _closest_equivalent_rotvec(
                pose[3:6],
                previous_rotvec,
                wrap_count=2,
            ).astype(np.float32, copy=False)
        pose_vectors[frame_index] = pose
        if np.isfinite(pose[3:6]).all():
            previous_rotvec = pose[3:6].astype(np.float64, copy=True)
    return pose_vectors


def _normalize_vector(vector, eps=1e-8):
    norm = float(np.linalg.norm(vector))
    if norm < eps:
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / norm).astype(np.float32, copy=False)


def _joint_indices(joint_count):
    if joint_count >= 21:
        return {
            "wrist": 0,
            "thumb_mcp": 1,
            "thumb_tip": 4,
            "index_mcp": 5,
            "index_tip": 8,
            "middle_mcp": 9,
            "middle_tip": 12,
            "ring_mcp": 13,
            "ring_tip": 16,
            "pinky_mcp": 17,
            "pinky_tip": 20,
        }
    if joint_count == 16:
        return {
            "wrist": 0,
            "index_mcp": 1,
            "index_tip": 3,
            "middle_mcp": 4,
            "middle_tip": 6,
            "ring_mcp": 7,
            "ring_tip": 9,
            "pinky_mcp": 10,
            "pinky_tip": 12,
            "thumb_mcp": 13,
            "thumb_tip": 15,
        }
    raise ValueError(f"unsupported joint_count for retarget: {joint_count}")


def _thumb_index_indices(joint_count):
    if joint_count >= 21:
        return [1, 2, 3, 4], [5, 6, 7, 8]
    if joint_count == 16:
        return [13, 14, 15], [1, 2, 3]
    raise ValueError(f"unsupported joint_count for retarget: {joint_count}")


def compute_pinch_norm(joints_cam, valid):
    points = ensure_float32(joints_cam)
    valid_mask = ensure_bool(valid).reshape(-1)
    thumb_idx, index_idx = _thumb_index_indices(points.shape[1])
    distances = np.zeros((points.shape[0],), dtype=np.float32)
    if points.shape[0] < 2:
        return np.zeros((points.shape[0],), dtype=np.float32)
    for frame_index in range(points.shape[0]):
        if frame_index >= valid_mask.shape[0] or not valid_mask[frame_index]:
            continue
        thumb_center = points[frame_index, thumb_idx].mean(axis=0)
        index_center = points[frame_index, index_idx].mean(axis=0)
        distance = np.linalg.norm(thumb_center - index_center)
        if np.isfinite(distance):
            distances[frame_index] = float(distance)

    closed_distance = PINCH_DIST_CLOSED_M
    open_distance = PINCH_DIST_OPEN_M
    span = open_distance - closed_distance
    if span <= 0:
        return np.zeros((points.shape[0],), dtype=np.float32)

    linear = np.clip((distances - closed_distance) / span, 0.0, 1.0)
    normed = np.power(linear, PINCH_GAMMA).astype(np.float32, copy=False)
    normed = np.nan_to_num(normed, nan=0.0, posinf=1.0, neginf=0.0)
    normed = np.clip(normed, 0.0, 1.0).astype(np.float32, copy=False)
    if valid_mask.shape[0] == points.shape[0]:
        normed[~valid_mask] = 0.0
    return normed


def _compute_palm_normal(joints, indices):
    wrist = joints[indices["wrist"]]
    index_mcp = joints[indices["index_mcp"]]
    pinky_mcp = joints[indices["pinky_mcp"]]
    normal = np.cross(index_mcp - wrist, pinky_mcp - wrist)
    normal = _normalize_vector(normal)
    if np.allclose(normal, 0.0):
        index_tip = joints[indices["index_tip"]]
        pinky_tip = joints[indices["pinky_tip"]]
        normal = _normalize_vector(np.cross(index_tip - wrist, pinky_tip - wrist))
    return normal


def _compute_cross_accumulated_y_axis(joints, wrist, thumb_idx, index_idx, side, *, normalize_inputs):
    near_count = min(2, len(thumb_idx), len(index_idx))
    y_sum = np.zeros(3, dtype=np.float32)
    for offset in range(near_count):
        thumb_vector = joints[thumb_idx[offset]] - wrist
        index_vector = joints[index_idx[offset]] - wrist
        if normalize_inputs:
            thumb_vector = _normalize_vector(thumb_vector)
            index_vector = _normalize_vector(index_vector)
        if side == "left":
            y_sum += np.cross(thumb_vector, index_vector)
        else:
            y_sum += np.cross(index_vector, thumb_vector)
    return _normalize_vector(y_sum)


def _finger_plane_weights(point_count):
    weights = np.ones((max(0, int(point_count)),), dtype=np.float32)
    if weights.size <= 0:
        return weights
    weights[0] = 2.0
    weights[-1] = max(float(weights[-1]), 2.0)
    return weights


def _test_finger_weights(point_count):
    weights = np.ones((max(0, int(point_count)),), dtype=np.float32)
    if weights.size <= 0:
        return weights
    weights[0] = 4.0
    weights[-1] = max(float(weights[-1]), 2.0)
    return weights


def _fit_weighted_plane_normal(points, weights):
    samples = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    sample_weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    if samples.shape[0] != sample_weights.shape[0] or samples.shape[0] < 3:
        return np.zeros((3,), dtype=np.float32)

    finite_mask = np.isfinite(samples).all(axis=1)
    finite_mask &= np.isfinite(sample_weights)
    finite_mask &= sample_weights > 0.0
    if int(np.count_nonzero(finite_mask)) < 3:
        return np.zeros((3,), dtype=np.float32)

    valid_samples = samples[finite_mask]
    valid_weights = sample_weights[finite_mask]
    weight_sum = float(valid_weights.sum())
    if weight_sum <= 1e-8:
        return np.zeros((3,), dtype=np.float32)

    center = (valid_samples * valid_weights.reshape(-1, 1)).sum(axis=0) / weight_sum
    centered = valid_samples - center.reshape(1, 3)
    covariance = (centered * valid_weights.reshape(-1, 1)).T @ centered
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance.astype(np.float64))
    except np.linalg.LinAlgError:
        return np.zeros((3,), dtype=np.float32)

    normal = eigenvectors[:, int(np.argmin(eigenvalues))]
    return _normalize_vector(normal.astype(np.float32, copy=False))


def _fit_weighted_line_to_points(points, weights):
    samples = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    sample_weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    if samples.shape[0] != sample_weights.shape[0] or samples.shape[0] < 2:
        return np.zeros((3,), dtype=np.float32), np.zeros((3,), dtype=np.float32)

    finite_mask = np.isfinite(samples).all(axis=1)
    finite_mask &= np.isfinite(sample_weights)
    finite_mask &= sample_weights > 0.0
    if int(np.count_nonzero(finite_mask)) < 2:
        return np.zeros((3,), dtype=np.float32), np.zeros((3,), dtype=np.float32)

    valid_samples = samples[finite_mask]
    valid_weights = sample_weights[finite_mask]
    weight_sum = float(valid_weights.sum())
    if weight_sum <= 1e-8:
        return np.zeros((3,), dtype=np.float32), np.zeros((3,), dtype=np.float32)

    center = (valid_samples * valid_weights.reshape(-1, 1)).sum(axis=0) / weight_sum
    centered = valid_samples - center.reshape(1, 3)
    covariance = (centered * valid_weights.reshape(-1, 1)).T @ centered
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance.astype(np.float64))
    except np.linalg.LinAlgError:
        return center.astype(np.float32, copy=False), np.zeros((3,), dtype=np.float32)

    direction = eigenvectors[:, int(np.argmax(eigenvalues))]
    return center.astype(np.float32, copy=False), _normalize_vector(direction.astype(np.float32, copy=False))


def _build_axis_orthogonal_basis(axis, *, preferred=None):
    direction = _normalize_vector(axis)
    if np.linalg.norm(direction) < 1e-8:
        return np.zeros((3,), dtype=np.float32), np.zeros((3,), dtype=np.float32)

    candidates: list[np.ndarray] = []
    if preferred is not None:
        candidates.append(np.asarray(preferred, dtype=np.float32).reshape(3))
    candidates.extend(
        [
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
        ]
    )

    basis_u = np.zeros((3,), dtype=np.float32)
    for candidate in candidates:
        basis_u = _normalize_vector(_reject_vector(candidate, direction))
        if np.linalg.norm(basis_u) >= 1e-8:
            break
    if np.linalg.norm(basis_u) < 1e-8:
        return np.zeros((3,), dtype=np.float32), np.zeros((3,), dtype=np.float32)

    basis_v = _normalize_vector(np.cross(direction, basis_u))
    if np.linalg.norm(basis_v) < 1e-8:
        return np.zeros((3,), dtype=np.float32), np.zeros((3,), dtype=np.float32)
    return basis_u, basis_v


def _fit_plane_normal_through_axis(points, weights, axis_point, axis_direction, *, preferred_normal=None):
    axis_origin = np.asarray(axis_point, dtype=np.float32).reshape(3)
    axis = _normalize_vector(axis_direction)
    if np.linalg.norm(axis) < 1e-8:
        return np.zeros((3,), dtype=np.float32)

    basis_u, basis_v = _build_axis_orthogonal_basis(axis, preferred=preferred_normal)
    if np.linalg.norm(basis_u) < 1e-8 or np.linalg.norm(basis_v) < 1e-8:
        return np.zeros((3,), dtype=np.float32)

    samples = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    sample_weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    if samples.shape[0] != sample_weights.shape[0] or samples.shape[0] < 2:
        return np.zeros((3,), dtype=np.float32)

    finite_mask = np.isfinite(samples).all(axis=1)
    finite_mask &= np.isfinite(sample_weights)
    finite_mask &= sample_weights > 0.0
    if int(np.count_nonzero(finite_mask)) < 2:
        return np.zeros((3,), dtype=np.float32)

    valid_samples = samples[finite_mask]
    valid_weights = sample_weights[finite_mask]
    centered = valid_samples - axis_origin.reshape(1, 3)
    residuals = np.stack(
        [
            centered @ basis_u,
            centered @ basis_v,
        ],
        axis=1,
    )
    covariance = (residuals * valid_weights.reshape(-1, 1)).T @ residuals
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance.astype(np.float64))
    except np.linalg.LinAlgError:
        return np.zeros((3,), dtype=np.float32)

    coeff = eigenvectors[:, int(np.argmin(eigenvalues))]
    normal = _normalize_vector(
        (float(coeff[0]) * basis_u + float(coeff[1]) * basis_v).astype(np.float32, copy=False)
    )
    if np.linalg.norm(normal) < 1e-8:
        return np.zeros((3,), dtype=np.float32)

    if preferred_normal is not None:
        preferred = _normalize_vector(preferred_normal)
        if np.linalg.norm(preferred) >= 1e-8 and float(np.dot(normal, preferred)) < 0.0:
            normal = -normal
    return normal.astype(np.float32, copy=False)


def _compute_weighted_plane_y_axis(joints, wrist, thumb_idx, index_idx, side):
    rough_y_axis = _compute_cross_accumulated_y_axis(
        joints,
        wrist,
        thumb_idx,
        index_idx,
        side,
        normalize_inputs=True,
    )

    thumb_points = np.asarray(joints[thumb_idx], dtype=np.float32).reshape(-1, 3)
    index_points = np.asarray(joints[index_idx], dtype=np.float32).reshape(-1, 3)
    plane_points = np.concatenate([thumb_points, index_points], axis=0)
    plane_weights = np.concatenate(
        [_finger_plane_weights(len(thumb_idx)), _finger_plane_weights(len(index_idx))],
        axis=0,
    )
    plane_normal = _fit_weighted_plane_normal(plane_points, plane_weights)
    if np.linalg.norm(plane_normal) < 1e-8:
        return rough_y_axis

    if np.linalg.norm(rough_y_axis) >= 1e-8 and float(np.dot(plane_normal, rough_y_axis)) < 0.0:
        plane_normal = -plane_normal
    return plane_normal.astype(np.float32, copy=False)


def _project_vector(vector, axis):
    base = np.asarray(vector, dtype=np.float32).reshape(3)
    direction = _normalize_vector(axis)
    if np.linalg.norm(direction) < 1e-8:
        return np.zeros((3,), dtype=np.float32)
    return (np.dot(base, direction) * direction).astype(np.float32, copy=False)


def _reject_vector(vector, axis):
    base = np.asarray(vector, dtype=np.float32).reshape(3)
    return (base - _project_vector(base, axis)).astype(np.float32, copy=False)


def _fit_line_to_points(points, *, anchor=None):
    samples = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if samples.shape[0] <= 0:
        return np.zeros((3,), dtype=np.float32), np.zeros((3,), dtype=np.float32)

    if anchor is None:
        center = samples.mean(axis=0)
    else:
        center = np.asarray(anchor, dtype=np.float32).reshape(3)
    centered = samples - center.reshape(1, 3)
    covariance = centered.T @ centered
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance.astype(np.float64))
    except np.linalg.LinAlgError:
        return center.astype(np.float32, copy=False), np.zeros((3,), dtype=np.float32)

    direction = eigenvectors[:, int(np.argmax(eigenvalues))]
    return center.astype(np.float32, copy=False), _normalize_vector(direction.astype(np.float32, copy=False))


def _project_point_onto_line(point, line_point, line_direction):
    target = np.asarray(point, dtype=np.float32).reshape(3)
    anchor = np.asarray(line_point, dtype=np.float32).reshape(3)
    direction = _normalize_vector(line_direction)
    if np.linalg.norm(direction) < 1e-8:
        return target.astype(np.float32, copy=False)
    return (anchor + np.dot(target - anchor, direction) * direction).astype(np.float32, copy=False)


def _compute_eef_pose_legacy_frame(joints, indices, thumb_idx, index_idx, side):
    wrist = joints[indices["wrist"]]
    thumb_points = joints[thumb_idx]
    index_points = joints[index_idx]
    thumb_center = thumb_points.mean(axis=0)
    index_center = index_points.mean(axis=0)
    origin = 0.5 * (joints[indices["index_tip"]] + joints[indices["middle_tip"]])

    y_axis = _compute_weighted_plane_y_axis(joints, wrist, thumb_idx, index_idx, side)

    if side == "right":
        x_raw = thumb_center - index_center
    else:
        x_raw = index_center - thumb_center
    x_raw = x_raw - np.dot(x_raw, y_axis) * y_axis
    x_axis = _normalize_vector(x_raw)
    if np.linalg.norm(x_axis) < 1e-8:
        x_axis = _normalize_vector(joints[indices["thumb_tip"]] - joints[indices["index_tip"]])
        x_axis = x_axis - np.dot(x_axis, y_axis) * y_axis
        x_axis = _normalize_vector(x_axis)

    z_axis = _normalize_vector(np.cross(x_axis, y_axis))
    if np.linalg.norm(y_axis) < 1e-8 or np.linalg.norm(x_axis) < 1e-8 or np.linalg.norm(z_axis) < 1e-8:
        palm_normal = _compute_palm_normal(joints, indices)
        x_axis = _normalize_vector(joints[indices["index_mcp"]] - wrist)
        y_axis = _normalize_vector(np.cross(palm_normal, x_axis))
        z_axis = _normalize_vector(np.cross(x_axis, y_axis))
        if np.linalg.norm(y_axis) < 1e-8 or np.linalg.norm(z_axis) < 1e-8:
            return None

    return origin, x_axis, y_axis, z_axis


def compute_eef_poses_legacy(joints_cam, valid, side):
    points = ensure_float32(joints_cam)
    valid_mask = ensure_bool(valid).reshape(-1)
    indices = _joint_indices(points.shape[1])
    thumb_idx, index_idx = _thumb_index_indices(points.shape[1])
    pinch = compute_pinch_norm(points, valid_mask)

    poses = np.full((points.shape[0], 7), np.nan, dtype=np.float32)
    for frame_index in range(points.shape[0]):
        if not valid_mask[frame_index]:
            continue

        joints = points[frame_index]
        pose_frame = _compute_eef_pose_legacy_frame(joints, indices, thumb_idx, index_idx, side)
        if pose_frame is None:
            continue
        origin, x_axis, y_axis, z_axis = pose_frame

        rotation = np.stack([x_axis, y_axis, z_axis], axis=1).astype(np.float32, copy=False)
        axis_angle = rotation_matrix_to_axis_angle(rotation)
        poses[frame_index, :3] = origin
        poses[frame_index, 3:6] = axis_angle
        poses[frame_index, 6] = pinch[frame_index]

    return poses


def _compute_eef_pose_pinch_plane_frame(joints, indices, thumb_idx, index_idx, side):
    wrist = joints[indices["wrist"]]
    thumb_tip = joints[indices["thumb_tip"]]
    index_tip = joints[indices["index_tip"]]
    origin = (0.5 * (thumb_tip + index_tip)).astype(np.float32, copy=False)

    thumb_line_point, thumb_line_direction = _fit_line_to_points(joints[thumb_idx], anchor=wrist)
    index_line_point, index_line_direction = _fit_line_to_points(joints[index_idx], anchor=wrist)
    projected_thumb_tip = _project_point_onto_line(thumb_tip, thumb_line_point, thumb_line_direction)
    projected_index_tip = _project_point_onto_line(index_tip, index_line_point, index_line_direction)
    y_axis = _compute_weighted_plane_y_axis(joints, wrist, thumb_idx, index_idx, side)
    if side == "left":
        x_raw = projected_index_tip - projected_thumb_tip
    else:
        x_raw = projected_thumb_tip - projected_index_tip
    x_raw = _reject_vector(x_raw, y_axis)
    x_axis = _normalize_vector(x_raw)
    z_axis = _normalize_vector(np.cross(x_axis, y_axis))

    if np.linalg.norm(x_axis) < 1e-8 or np.linalg.norm(y_axis) < 1e-8 or np.linalg.norm(z_axis) < 1e-8:
        pose_frame = _compute_eef_pose_legacy_frame(joints, indices, thumb_idx, index_idx, side)
        if pose_frame is None:
            return None
        _, x_axis, y_axis, z_axis = pose_frame

    return origin, x_axis, y_axis, z_axis


def _compute_eef_pose_test_frame(joints, indices, thumb_idx, index_idx, side):
    wrist = joints[indices["wrist"]]
    thumb_points = np.asarray(joints[thumb_idx], dtype=np.float32).reshape(-1, 3)
    index_points = np.asarray(joints[index_idx], dtype=np.float32).reshape(-1, 3)
    thumb_weights = _test_finger_weights(len(thumb_idx))
    index_weights = _test_finger_weights(len(index_idx))

    thumb_line_point, thumb_line_direction = _fit_weighted_line_to_points(thumb_points, thumb_weights)
    index_line_point, index_line_direction = _fit_weighted_line_to_points(index_points, index_weights)
    if np.linalg.norm(thumb_line_direction) < 1e-8 or np.linalg.norm(index_line_direction) < 1e-8:
        return _compute_eef_pose_pinch_plane_frame(joints, indices, thumb_idx, index_idx, side)

    thumb_projected = np.stack(
        [
            _project_point_onto_line(point, thumb_line_point, thumb_line_direction)
            for point in thumb_points
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    index_projected = np.stack(
        [
            _project_point_onto_line(point, index_line_point, index_line_direction)
            for point in index_points
        ],
        axis=0,
    ).astype(np.float32, copy=False)

    paired_count = min(3, thumb_projected.shape[0], index_projected.shape[0])
    if paired_count <= 0:
        return _compute_eef_pose_pinch_plane_frame(joints, indices, thumb_idx, index_idx, side)
    selected_thumb_projected = thumb_projected[:paired_count]
    selected_index_projected = index_projected[:paired_count]
    pair_midpoints = 0.5 * (selected_thumb_projected + selected_index_projected)
    if side == "left":
        pair_vectors = selected_index_projected - selected_thumb_projected
    else:
        pair_vectors = selected_thumb_projected - selected_index_projected
    pair_norms = np.linalg.norm(pair_vectors, axis=1)
    finite_pair_mask = np.isfinite(pair_vectors).all(axis=1) & np.isfinite(pair_midpoints).all(axis=1) & np.isfinite(pair_norms)
    if int(np.count_nonzero(finite_pair_mask)) <= 0:
        return _compute_eef_pose_pinch_plane_frame(joints, indices, thumb_idx, index_idx, side)
    # The pose stores a unit x-axis, so we average the first three corresponding
    # thumb-index connection vectors and then normalize the result.
    x_axis = _normalize_vector(pair_vectors[finite_pair_mask].mean(axis=0))
    if np.linalg.norm(x_axis) < 1e-8:
        return _compute_eef_pose_pinch_plane_frame(joints, indices, thumb_idx, index_idx, side)

    rough_y_axis = _compute_cross_accumulated_y_axis(
        joints,
        wrist,
        thumb_idx,
        index_idx,
        side,
        normalize_inputs=True,
    )
    plane_points = np.concatenate([thumb_projected, index_projected], axis=0)
    plane_weights = np.concatenate([thumb_weights, index_weights], axis=0)
    x_axis_origin = pair_midpoints[finite_pair_mask].mean(axis=0).astype(np.float32, copy=False)
    y_axis = _fit_plane_normal_through_axis(
        plane_points,
        plane_weights,
        x_axis_origin,
        x_axis,
        preferred_normal=rough_y_axis,
    )
    if np.linalg.norm(y_axis) < 1e-8:
        return _compute_eef_pose_pinch_plane_frame(joints, indices, thumb_idx, index_idx, side)

    z_axis = _normalize_vector(np.cross(x_axis, y_axis))
    if np.linalg.norm(z_axis) < 1e-8:
        return _compute_eef_pose_pinch_plane_frame(joints, indices, thumb_idx, index_idx, side)

    # The test scheme keeps the position defined by the midpoint of thumb/index tips.
    origin = (0.5 * (joints[indices["thumb_tip"]] + joints[indices["index_tip"]])).astype(np.float32, copy=False)
    return origin, x_axis, y_axis, z_axis


def _compute_pose_sequence_with_frame_builder(joints_cam, valid, side, frame_builder):
    points = ensure_float32(joints_cam)
    valid_mask = ensure_bool(valid).reshape(-1)
    indices = _joint_indices(points.shape[1])
    thumb_idx, index_idx = _thumb_index_indices(points.shape[1])
    pinch = compute_pinch_norm(points, valid_mask)

    poses = np.full((points.shape[0], 7), np.nan, dtype=np.float32)
    for frame_index in range(points.shape[0]):
        if not valid_mask[frame_index]:
            continue

        joints = points[frame_index]
        pose_frame = frame_builder(joints, indices, thumb_idx, index_idx, side)
        if pose_frame is None:
            continue
        origin, x_axis, y_axis, z_axis = pose_frame

        rotation = np.stack([x_axis, y_axis, z_axis], axis=1).astype(np.float32, copy=False)
        axis_angle = rotation_matrix_to_axis_angle(rotation)
        poses[frame_index, :3] = origin
        poses[frame_index, 3:6] = axis_angle
        poses[frame_index, 6] = pinch[frame_index]

    return poses


def compute_eef_poses_pinch_plane(joints_cam, valid, side):
    return _compute_pose_sequence_with_frame_builder(
        joints_cam,
        valid,
        side,
        _compute_eef_pose_pinch_plane_frame,
    )


def compute_eef_poses_test(joints_cam, valid, side):
    return _compute_pose_sequence_with_frame_builder(
        joints_cam,
        valid,
        side,
        _compute_eef_pose_test_frame,
    )


def compute_eef_poses(joints_cam, valid, side, scheme="legacy"):
    scheme_name = str(scheme or "legacy").strip().lower()
    if scheme_name == "legacy":
        return compute_eef_poses_legacy(joints_cam, valid, side)
    if scheme_name == "pinch_plane":
        return compute_eef_poses_pinch_plane(joints_cam, valid, side)
    if scheme_name == "test":
        return compute_eef_poses_test(joints_cam, valid, side)
    raise ValueError(f"unsupported retarget scheme: {scheme}")
