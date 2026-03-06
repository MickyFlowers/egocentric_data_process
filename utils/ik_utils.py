from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation as Rscipy

try:
    import pinocchio as pin
except Exception:  # pragma: no cover - handled at runtime when IK process is used.
    pin = None


@dataclass
class IKConfig:
    position_weight: float = 8.0
    orientation_weight: float = 1.0
    continuity_weight: float = 0.05
    damping: float = 1.0e-4
    max_iterations: int = 64
    max_joint_step: float = 0.15
    max_frame_delta: float = 0.35
    position_tolerance: float = 0.02
    orientation_tolerance: float = 0.25
    early_stop_position_tolerance: float | None = None
    early_stop_orientation_tolerance: float | None = None
    collision_weight: float = 5.0
    collision_safe_distance: float = 0.01
    collision_check_interval: int = 2
    collision_filter_adjacent_pairs: bool = True
    collision_filter_neutral_touching_pairs: bool = True
    collision_neutral_touching_tolerance: float = 1.0e-9
    collision_ignore_links: tuple[str, ...] = ()
    collision_force_include_link_pairs: tuple[tuple[str, str], ...] = ()
    use_collision: bool = False
    require_collision: bool = False
    use_joint_ekf_smoothing: bool = True
    ekf_dt: float = 1.0
    ekf_process_noise_position: float = 1.2e-4
    ekf_process_noise_velocity: float = 4.0e-3
    ekf_measurement_noise: float = 4.0e-3
    ekf_initial_position_var: float = 2.0e-4
    ekf_initial_velocity_var: float = 0.05


def ensure_pose_array(values: Any) -> np.ndarray:
    poses = np.asarray(values, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] < 6:
        raise ValueError("retarget poses must be shaped as [N, >=6] (xyz + axis_angle)")
    return poses


def ensure_sample_array(values: Any, *, count: int) -> np.ndarray:
    if values is None:
        return np.zeros((count,), dtype=np.float64)
    sample = np.asarray(values, dtype=np.float64).reshape(-1)
    if sample.shape[0] < count:
        padded = np.zeros((count,), dtype=np.float64)
        padded[: sample.shape[0]] = sample
        sample = padded
    elif sample.shape[0] > count:
        sample = sample[:count]
    sample = np.nan_to_num(sample, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(sample, 0.0, 1.0)


def map_gripper_samples_to_joint_targets(
    sample: np.ndarray,
    *,
    closed_positions: list[float],
    open_positions: list[float],
) -> np.ndarray:
    if len(closed_positions) != len(open_positions):
        raise ValueError("closed_positions and open_positions must have the same length")
    closed = np.asarray(closed_positions, dtype=np.float64).reshape(1, -1)
    opened = np.asarray(open_positions, dtype=np.float64).reshape(1, -1)
    ratio = ensure_sample_array(sample, count=sample.shape[0]).reshape(-1, 1)
    return closed + ratio * (opened - closed)


class PinocchioIKSolver:
    """Sequential damped-least-squares IK with joint-limit projection and continuity regularization."""

    def __init__(
        self,
        *,
        urdf_path: str | Path,
        package_dirs: list[str] | None,
        ee_frame: str,
        joint_names: list[str],
        base_frame: str | None = None,
        config: IKConfig | None = None,
    ) -> None:
        if pin is None:
            raise ModuleNotFoundError(
                "pinocchio is required by InverseKinematicsProcess. "
                "Install it before running IK (e.g. conda install -c conda-forge pinocchio hpp-fcl)."
            )
        if not joint_names:
            raise ValueError("joint_names cannot be empty")

        self.config = config or IKConfig()
        self.ignored_collision_links = {str(name) for name in self.config.collision_ignore_links if str(name)}
        self.forced_collision_link_pairs = self._normalize_link_pair_set(
            self.config.collision_force_include_link_pairs
        )
        self.early_stop_position_tolerance = float(
            self.config.early_stop_position_tolerance
            if self.config.early_stop_position_tolerance is not None
            else self.config.position_tolerance
        )
        self.early_stop_orientation_tolerance = float(
            self.config.early_stop_orientation_tolerance
            if self.config.early_stop_orientation_tolerance is not None
            else self.config.orientation_tolerance
        )
        self.urdf_path = Path(urdf_path).expanduser().resolve()
        self.package_dirs = [str(Path(path).expanduser().resolve()) for path in (package_dirs or [])]
        self.model = pin.buildModelFromUrdf(str(self.urdf_path))
        self.data = self.model.createData()
        self.adjacent_collision_link_pairs = self._build_adjacent_collision_link_pairs()

        self.ee_frame_id = self._require_frame_id(ee_frame)
        self.base_frame_id = self._require_frame_id(base_frame) if base_frame else None

        self.joint_names = list(joint_names)
        self.joint_ids = [self._require_joint_id(name) for name in self.joint_names]
        self.q_indices: list[int] = []
        self.v_indices: list[int] = []
        self._joint_q_index_cache: dict[str, int] = {}
        for joint_id in self.joint_ids:
            if int(self.model.nqs[joint_id]) != 1 or int(self.model.nvs[joint_id]) != 1:
                joint_name = self.model.names[joint_id]
                raise ValueError(
                    f"joint '{joint_name}' is not 1-DoF (nq={self.model.nqs[joint_id]}, nv={self.model.nvs[joint_id]})."
                )
            self.q_indices.append(int(self.model.idx_qs[joint_id]))
            self.v_indices.append(int(self.model.idx_vs[joint_id]))
        self.q_indices_arr = np.asarray(self.q_indices, dtype=np.int64)
        self.v_indices_arr = np.asarray(self.v_indices, dtype=np.int64)

        full_lower = self.model.lowerPositionLimit.astype(np.float64, copy=True)
        full_upper = self.model.upperPositionLimit.astype(np.float64, copy=True)
        full_lower[~np.isfinite(full_lower)] = -np.inf
        full_upper[~np.isfinite(full_upper)] = np.inf
        self.full_lower = full_lower
        self.full_upper = full_upper

        lower = self.full_lower[self.q_indices_arr].copy()
        upper = self.full_upper[self.q_indices_arr].copy()
        lower[~np.isfinite(lower)] = -np.inf
        upper[~np.isfinite(upper)] = np.inf
        self.lower = lower
        self.upper = upper
        self.has_lower = np.isfinite(self.lower)
        self.has_upper = np.isfinite(self.upper)

        neutral = pin.neutral(self.model).astype(np.float64, copy=True)
        self.neutral = self._clip_q(neutral)

        pin.forwardKinematics(self.model, self.data, self.neutral)
        pin.updateFramePlacements(self.model, self.data)
        self.world_from_base = None
        if self.base_frame_id is not None:
            self.world_from_base = self.data.oMf[self.base_frame_id].copy()

        self.collision_model = None
        self.collision_data = None
        self.collision_pair_mask = np.zeros((0,), dtype=np.bool_)
        self.collision_pair_count_total = 0
        self.collision_pair_count_active = 0
        self.collision_pair_count_filtered = 0
        self.collision_available = False
        self.collision_enabled = False
        self._try_enable_collision()

    def solve_trajectory(
        self,
        *,
        pose_vectors: np.ndarray,
        initial_joint_positions: list[float] | np.ndarray | None = None,
        compute_collision: bool = True,
    ) -> dict[str, Any]:
        poses = ensure_pose_array(pose_vectors)
        frame_count = int(poses.shape[0])
        q = self.neutral.copy()
        if initial_joint_positions is not None:
            active_init = np.asarray(initial_joint_positions, dtype=np.float64).reshape(-1)
            if active_init.shape[0] != len(self.joint_names):
                raise ValueError(
                    f"initial_joint_positions length mismatch: expected {len(self.joint_names)}, got {active_init.shape[0]}"
                )
            q[self.q_indices_arr] = active_init
            q = self._clip_q(q)

        q_prev = q[self.q_indices_arr].copy()
        joints = np.zeros((frame_count, len(self.joint_names)), dtype=np.float64)
        position_error = np.full((frame_count,), np.nan, dtype=np.float64)
        orientation_error = np.full((frame_count,), np.nan, dtype=np.float64)
        iterations = np.zeros((frame_count,), dtype=np.int32)
        reachable = np.zeros((frame_count,), dtype=np.bool_)

        for frame_index in range(frame_count):
            frame_pose = poses[frame_index, :6]
            if not np.isfinite(frame_pose).all():
                joints[frame_index] = q_prev
                continue

            target = self._target_from_pose(frame_pose)
            q, metrics = self._solve_single_pose(target=target, q_seed=q, q_prev=q_prev)

            q_active = q[self.q_indices_arr].copy()
            if self.config.max_frame_delta > 0.0:
                clipped_delta = np.clip(
                    q_active - q_prev,
                    -self.config.max_frame_delta,
                    self.config.max_frame_delta,
                )
                q_active = q_prev + clipped_delta
                q[self.q_indices_arr] = q_active
                q = self._clip_q(q)
                metrics = self._frame_metrics(q=q, target=target, q_prev=q_prev, iteration=metrics["iteration"])
                q_active = q[self.q_indices_arr].copy()

            joints[frame_index] = q_active
            q_prev = q_active
            position_error[frame_index] = metrics["position_error"]
            orientation_error[frame_index] = metrics["orientation_error"]
            iterations[frame_index] = int(metrics["iteration"])
            reachable[frame_index] = bool(
                metrics["position_error"] <= self.config.position_tolerance
                and metrics["orientation_error"] <= self.config.orientation_tolerance
            )

        self._retry_unreachable_frames(
            poses=poses[:, :6],
            joints=joints,
            position_error=position_error,
            orientation_error=orientation_error,
            iterations=iterations,
            reachable=reachable,
        )
        ekf_applied = bool(self.config.use_joint_ekf_smoothing and frame_count > 1)
        if ekf_applied:
            joints = self._smooth_joint_trajectory_with_ekf(joints)
        # Always recompute from final returned joints to keep metrics strictly aligned
        # with the trajectory emitted by solver (post-EKF when enabled).
        self._recompute_errors_from_joints(
            poses=poses[:, :6],
            joints=joints,
            position_error=position_error,
            orientation_error=orientation_error,
            reachable=reachable,
        )
        collision = self._compute_collision_flags_from_joints(joints) if compute_collision else None

        return {
            "joint_names": list(self.joint_names),
            "joint_positions": joints,
            "position_error": position_error,
            "orientation_error": orientation_error,
            "iterations": iterations,
            "reachable": reachable,
            "collision": collision,
            "ekf_applied": ekf_applied,
        }

    def _solve_single_pose(self, *, target: Any, q_seed: np.ndarray, q_prev: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        current_q = q_seed.copy()
        current = self._frame_metrics(q=current_q, target=target, q_prev=q_prev, iteration=0)
        best_q = current_q.copy()
        best = dict(current)
        last_iteration = 0

        for iteration in range(1, self.config.max_iterations + 1):
            last_iteration = iteration
            if (
                current["position_error"] <= self.early_stop_position_tolerance
                and current["orientation_error"] <= self.early_stop_orientation_tolerance
            ):
                break
            err = current["error_vector"]

            jacobian = pin.computeFrameJacobian(
                self.model,
                self.data,
                current_q,
                self.ee_frame_id,
                pin.ReferenceFrame.LOCAL,
            )
            jacobian_active = jacobian[:, self.v_indices_arr]

            weight = np.array(
                [
                    self.config.position_weight,
                    self.config.position_weight,
                    self.config.position_weight,
                    self.config.orientation_weight,
                    self.config.orientation_weight,
                    self.config.orientation_weight,
                ],
                dtype=np.float64,
            )
            w_sqrt = np.sqrt(weight)
            jw = jacobian_active * w_sqrt[:, None]
            ew = err * w_sqrt

            q_active = current_q[self.q_indices_arr]
            smooth_grad = self.config.continuity_weight * (q_prev - q_active)
            hessian = (
                jw.T @ jw
                + (self.config.damping + self.config.continuity_weight) * np.eye(jacobian_active.shape[1], dtype=np.float64)
            )
            gradient = jw.T @ ew + smooth_grad
            try:
                dq_active = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                dq_active = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
            dq_active = np.clip(dq_active, -self.config.max_joint_step, self.config.max_joint_step)

            next_q, next_metrics = self._line_search(
                q=current_q,
                dq_active=dq_active,
                target=target,
                q_prev=q_prev,
                iteration=iteration,
            )
            if np.allclose(next_q, current_q, rtol=0.0, atol=1.0e-10):
                current_q = next_q
                current = next_metrics
                if current["objective"] + 1.0e-12 < best["objective"]:
                    best_q = current_q.copy()
                    best = dict(current)
                break

            current_q = next_q
            current = next_metrics
            if current["objective"] + 1.0e-12 < best["objective"]:
                best_q = current_q.copy()
                best = dict(current)

        # Report metrics that are strictly aligned with the returned best_q.
        result = dict(best)
        result["iteration"] = int(last_iteration)
        return best_q, result

    def _line_search(
        self,
        *,
        q: np.ndarray,
        dq_active: np.ndarray,
        target: Any,
        q_prev: np.ndarray,
        iteration: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        best_q = q
        best_metrics = self._frame_metrics(q=q, target=target, q_prev=q_prev, iteration=iteration)
        for direction in (1.0, -1.0):
            for alpha in (1.0, 0.5, 0.25, 0.125):
                dq_full = np.zeros((self.model.nv,), dtype=np.float64)
                dq_full[self.v_indices_arr] = direction * alpha * dq_active
                q_candidate = pin.integrate(self.model, q, dq_full)
                q_candidate = self._clip_q(q_candidate)
                metrics = self._frame_metrics(q=q_candidate, target=target, q_prev=q_prev, iteration=iteration)
                if metrics["objective"] + 1.0e-12 < best_metrics["objective"]:
                    best_q = q_candidate
                    best_metrics = metrics
        return best_q, best_metrics

    def _frame_metrics(self, *, q: np.ndarray, target: Any, q_prev: np.ndarray, iteration: int) -> dict[str, Any]:
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        current = self.data.oMf[self.ee_frame_id]
        delta = current.actInv(target)
        error_vector = pin.log(delta).vector.astype(np.float64, copy=False)
        # Pinocchio log(SE3) vector layout is [translation(3), rotation(3)].
        position_error = float(np.linalg.norm(error_vector[:3]))
        orientation_error = float(np.linalg.norm(error_vector[3:6]))

        q_active = q[self.q_indices_arr]
        smooth_penalty = float(np.dot(q_active - q_prev, q_active - q_prev))
        objective = (
            self.config.position_weight * position_error * position_error
            + self.config.orientation_weight * orientation_error * orientation_error
            + self.config.continuity_weight * smooth_penalty
        )
        if self.collision_enabled and (iteration % max(1, self.config.collision_check_interval) == 0):
            objective += self.config.collision_weight * self._collision_penalty(q)

        return {
            "objective": float(objective),
            "position_error": position_error,
            "orientation_error": orientation_error,
            "error_vector": error_vector,
            "iteration": int(iteration),
        }

    def smooth_trajectory_with_ekf(
        self,
        trajectory: np.ndarray,
        *,
        lower: np.ndarray | None = None,
        upper: np.ndarray | None = None,
    ) -> np.ndarray:
        q_obs = np.asarray(trajectory, dtype=np.float64)
        if q_obs.ndim != 2:
            raise ValueError(f"trajectory shape mismatch: expected [N, D], got {q_obs.shape}")
        if q_obs.shape[0] <= 0:
            return q_obs.copy()

        frame_count, joint_count = q_obs.shape
        state_dim = joint_count * 2
        dt = float(max(self.config.ekf_dt, 1.0e-6))

        transition = np.eye(state_dim, dtype=np.float64)
        transition[:joint_count, joint_count:] = dt * np.eye(joint_count, dtype=np.float64)
        observation = np.zeros((joint_count, state_dim), dtype=np.float64)
        observation[:, :joint_count] = np.eye(joint_count, dtype=np.float64)

        process_noise = np.zeros((state_dim, state_dim), dtype=np.float64)
        process_noise[:joint_count, :joint_count] = (
            self.config.ekf_process_noise_position * np.eye(joint_count, dtype=np.float64)
        )
        process_noise[joint_count:, joint_count:] = (
            self.config.ekf_process_noise_velocity * np.eye(joint_count, dtype=np.float64)
        )
        measurement_noise = self.config.ekf_measurement_noise * np.eye(joint_count, dtype=np.float64)

        filtered_state = np.zeros((frame_count, state_dim), dtype=np.float64)
        filtered_cov = np.zeros((frame_count, state_dim, state_dim), dtype=np.float64)
        predicted_state = np.zeros((frame_count, state_dim), dtype=np.float64)
        predicted_cov = np.zeros((frame_count, state_dim, state_dim), dtype=np.float64)

        state = np.zeros((state_dim,), dtype=np.float64)
        state[:joint_count] = q_obs[0]
        if frame_count > 1:
            state[joint_count:] = (q_obs[1] - q_obs[0]) / dt
        covariance = np.zeros((state_dim, state_dim), dtype=np.float64)
        covariance[:joint_count, :joint_count] = (
            self.config.ekf_initial_position_var * np.eye(joint_count, dtype=np.float64)
        )
        covariance[joint_count:, joint_count:] = (
            self.config.ekf_initial_velocity_var * np.eye(joint_count, dtype=np.float64)
        )
        identity = np.eye(state_dim, dtype=np.float64)

        for frame_index in range(frame_count):
            state_pred = transition @ state
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                cov_pred_raw = transition @ covariance @ transition.T + process_noise
            cov_pred = self._stabilize_covariance(cov_pred_raw)

            predicted_state[frame_index] = state_pred
            predicted_cov[frame_index] = cov_pred

            measurement = q_obs[frame_index]
            innovation = measurement - observation @ state_pred
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                innovation_cov = observation @ cov_pred @ observation.T + measurement_noise
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                kalman_rhs = (cov_pred @ observation.T).T
            kalman_rhs = np.nan_to_num(kalman_rhs, nan=0.0, posinf=0.0, neginf=0.0)
            try:
                kalman_gain = np.linalg.solve(innovation_cov.T, kalman_rhs).T
            except np.linalg.LinAlgError:
                kalman_gain = (cov_pred @ observation.T) @ np.linalg.pinv(innovation_cov)
            kalman_gain = np.nan_to_num(kalman_gain, nan=0.0, posinf=0.0, neginf=0.0)
            kalman_gain = np.clip(kalman_gain, -1.0e3, 1.0e3)
            state = state_pred + kalman_gain @ innovation
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                correction = identity - kalman_gain @ observation
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                covariance_raw = correction @ cov_pred @ correction.T + kalman_gain @ measurement_noise @ kalman_gain.T
            covariance = self._stabilize_covariance(covariance_raw)
            filtered_state[frame_index] = state
            filtered_cov[frame_index] = covariance

        smoothed_state = filtered_state.copy()
        smoothed_cov = filtered_cov.copy()
        for frame_index in range(frame_count - 2, -1, -1):
            next_cov_pred = self._stabilize_covariance(predicted_cov[frame_index + 1])
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                rts_rhs = (filtered_cov[frame_index] @ transition.T).T
            rts_rhs = np.nan_to_num(rts_rhs, nan=0.0, posinf=0.0, neginf=0.0)
            try:
                gain = np.linalg.solve(next_cov_pred.T, rts_rhs).T
            except np.linalg.LinAlgError:
                gain = (filtered_cov[frame_index] @ transition.T) @ np.linalg.pinv(next_cov_pred)
            gain = np.nan_to_num(gain, nan=0.0, posinf=0.0, neginf=0.0)
            gain = np.clip(gain, -1.0e3, 1.0e3)

            smoothed_state[frame_index] = filtered_state[frame_index] + gain @ (
                smoothed_state[frame_index + 1] - predicted_state[frame_index + 1]
            )
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                smoothed_cov_raw = filtered_cov[frame_index] + gain @ (
                    smoothed_cov[frame_index + 1] - next_cov_pred
                ) @ gain.T
            smoothed_cov[frame_index] = self._stabilize_covariance(smoothed_cov_raw)

        smoothed_q = smoothed_state[:, :joint_count]
        return self._clip_trajectory_to_bounds(
            smoothed_q,
            lower=lower,
            upper=upper,
            expected_dim=joint_count,
        )

    def _smooth_joint_trajectory_with_ekf(self, joints: np.ndarray) -> np.ndarray:
        return self.smooth_trajectory_with_ekf(
            joints,
            lower=self.lower,
            upper=self.upper,
        )

    @staticmethod
    def _stabilize_covariance(covariance: np.ndarray, limit: float = 1.0e6) -> np.ndarray:
        cov = np.asarray(covariance, dtype=np.float64)
        cov = 0.5 * (cov + cov.T)
        cov = np.nan_to_num(cov, nan=limit, posinf=limit, neginf=-limit)
        return np.clip(cov, -limit, limit)

    def _clip_joint_trajectory(self, joint_positions: np.ndarray) -> np.ndarray:
        return self._clip_trajectory_to_bounds(
            joint_positions,
            lower=self.lower,
            upper=self.upper,
            expected_dim=len(self.q_indices),
        )

    @staticmethod
    def _clip_trajectory_to_bounds(
        trajectory: np.ndarray,
        *,
        lower: np.ndarray | None = None,
        upper: np.ndarray | None = None,
        expected_dim: int | None = None,
    ) -> np.ndarray:
        clipped = np.asarray(trajectory, dtype=np.float64).copy()
        if clipped.ndim != 2:
            raise ValueError(f"trajectory shape mismatch: expected [N, D], got {clipped.shape}")
        dim = int(clipped.shape[1])
        if expected_dim is not None and dim != int(expected_dim):
            raise ValueError(f"trajectory dim mismatch: expected D={expected_dim}, got D={dim}")

        if lower is not None:
            lower_arr = np.asarray(lower, dtype=np.float64).reshape(-1)
            if lower_arr.shape[0] != dim:
                raise ValueError(
                    f"lower bound dim mismatch: expected D={dim}, got D={lower_arr.shape[0]}"
                )
            finite_lower = np.isfinite(lower_arr)
            if finite_lower.any():
                clipped[:, finite_lower] = np.maximum(
                    clipped[:, finite_lower],
                    lower_arr[finite_lower].reshape(1, -1),
                )

        if upper is not None:
            upper_arr = np.asarray(upper, dtype=np.float64).reshape(-1)
            if upper_arr.shape[0] != dim:
                raise ValueError(
                    f"upper bound dim mismatch: expected D={dim}, got D={upper_arr.shape[0]}"
                )
            finite_upper = np.isfinite(upper_arr)
            if finite_upper.any():
                clipped[:, finite_upper] = np.minimum(
                    clipped[:, finite_upper],
                    upper_arr[finite_upper].reshape(1, -1),
                )
        return clipped

    def _recompute_errors_from_joints(
        self,
        *,
        poses: np.ndarray,
        joints: np.ndarray,
        position_error: np.ndarray,
        orientation_error: np.ndarray,
        reachable: np.ndarray,
    ) -> None:
        frame_count = int(joints.shape[0])
        finite_pose_mask = np.isfinite(poses).all(axis=1)
        for frame_index in range(frame_count):
            if not finite_pose_mask[frame_index]:
                position_error[frame_index] = np.nan
                orientation_error[frame_index] = np.nan
                reachable[frame_index] = False
                continue
            q = self.neutral.copy()
            q[self.q_indices_arr] = joints[frame_index]
            q = self._clip_q(q)
            target = self._target_from_pose(poses[frame_index])
            metrics = self._frame_metrics(
                q=q,
                target=target,
                q_prev=joints[max(frame_index - 1, 0)],
                iteration=0,
            )
            position_error[frame_index] = metrics["position_error"]
            orientation_error[frame_index] = metrics["orientation_error"]
            reachable[frame_index] = bool(
                metrics["position_error"] <= self.config.position_tolerance
                and metrics["orientation_error"] <= self.config.orientation_tolerance
            )

    def _compute_collision_flags_from_joints(self, joints: np.ndarray) -> np.ndarray | None:
        if not self.collision_available:
            return None

        frame_count = int(joints.shape[0])
        collision = np.zeros((frame_count,), dtype=np.bool_)
        for frame_index in range(frame_count):
            q = self.neutral.copy()
            q[self.q_indices_arr] = joints[frame_index]
            q = self._clip_q(q)
            collision[frame_index] = self._evaluate_collision(q)[1]
        return collision

    def compute_collision_flags_from_joint_state_trajectory(
        self,
        *,
        joint_names: list[str],
        joint_positions: np.ndarray,
    ) -> np.ndarray | None:
        if not self.collision_available:
            return None

        names = [str(name) for name in joint_names]
        positions = np.asarray(joint_positions, dtype=np.float64)
        if positions.ndim != 2:
            raise ValueError(f"joint_positions shape mismatch: expected [N, D], got {positions.shape}")
        if positions.shape[1] != len(names):
            raise ValueError(
                f"joint_positions dim mismatch: expected D={len(names)}, got D={positions.shape[1]}"
            )

        frame_count = int(positions.shape[0])
        if frame_count <= 0:
            return np.zeros((0,), dtype=np.bool_)

        q_indices = self._q_indices_for_joint_names(names)
        lower = self.full_lower[q_indices]
        upper = self.full_upper[q_indices]
        collision = np.zeros((frame_count,), dtype=np.bool_)
        for frame_index in range(frame_count):
            q = self.neutral.copy()
            if q_indices.size > 0:
                q[q_indices] = positions[frame_index]
                q = self._clip_q_indices(q, q_indices=q_indices, lower=lower, upper=upper)
            collision[frame_index] = self._evaluate_collision(q)[1]
        return collision

    def _retry_unreachable_frames(
        self,
        *,
        poses: np.ndarray,
        joints: np.ndarray,
        position_error: np.ndarray,
        orientation_error: np.ndarray,
        iterations: np.ndarray,
        reachable: np.ndarray,
    ) -> None:
        unreachable_indices = np.flatnonzero(~reachable)
        if unreachable_indices.size == 0:
            return

        reachable_indices = np.flatnonzero(reachable)
        if reachable_indices.size == 0:
            return

        finite_pose_mask = np.isfinite(poses).all(axis=1)
        for frame_index in unreachable_indices:
            if not finite_pose_mask[frame_index]:
                continue
            if reachable_indices.size == 0:
                break

            nearest_reachable_index = int(
                reachable_indices[np.argmin(np.abs(reachable_indices - frame_index))]
            )
            q_seed = self.neutral.copy()
            q_seed[self.q_indices_arr] = joints[nearest_reachable_index]
            q_seed = self._clip_q(q_seed)

            target = self._target_from_pose(poses[frame_index])
            q_candidate, metrics = self._solve_single_pose(
                target=target,
                q_seed=q_seed,
                q_prev=joints[nearest_reachable_index],
            )

            old_position_error = position_error[frame_index]
            old_orientation_error = orientation_error[frame_index]
            better_position = (
                not np.isfinite(old_position_error)
                or metrics["position_error"] < old_position_error - 1.0e-9
            )
            better_orientation = (
                not np.isfinite(old_orientation_error)
                or metrics["orientation_error"] < old_orientation_error - 1.0e-9
            )
            if not (better_position or better_orientation):
                continue

            joints[frame_index] = q_candidate[self.q_indices_arr].copy()
            position_error[frame_index] = metrics["position_error"]
            orientation_error[frame_index] = metrics["orientation_error"]
            iterations[frame_index] = int(iterations[frame_index] + metrics["iteration"])
            reachable[frame_index] = bool(
                metrics["position_error"] <= self.config.position_tolerance
                and metrics["orientation_error"] <= self.config.orientation_tolerance
            )
            if reachable[frame_index]:
                reachable_indices = np.sort(np.append(reachable_indices, frame_index))

    def _collision_penalty(self, q: np.ndarray) -> float:
        if not self.collision_enabled:
            return 0.0
        return self._evaluate_collision(q)[0]

    def _evaluate_collision(self, q: np.ndarray) -> tuple[float, bool]:
        if not self.collision_available or self.collision_model is None or self.collision_data is None:
            return 0.0, False
        try:
            self._update_collision_distances(q)
        except Exception:
            return 0.0, False

        safe_distance = float(self.config.collision_safe_distance)
        penalty = 0.0
        in_collision = False
        for result in self.collision_data.distanceResults:
            distance = getattr(result, "min_distance", None)
            if distance is None:
                distance = getattr(result, "minDistance", None)
            if distance is None:
                continue
            value = float(distance)
            if not np.isfinite(value):
                continue
            if value < safe_distance:
                gap = safe_distance - value
                penalty += gap * gap
                in_collision = True
        return float(penalty), bool(in_collision)

    def _target_from_pose(self, pose: np.ndarray):
        translation = np.asarray(pose[:3], dtype=np.float64)
        rotation = Rscipy.from_rotvec(np.asarray(pose[3:6], dtype=np.float64)).as_matrix().astype(np.float64)
        target = pin.SE3(rotation, translation)
        if self.world_from_base is None:
            return target
        return self.world_from_base * target

    def _clip_q(self, q: np.ndarray) -> np.ndarray:
        return self._clip_q_indices(
            q,
            q_indices=self.q_indices_arr,
            lower=self.lower,
            upper=self.upper,
        )

    def _try_enable_collision(self) -> None:
        try:
            self.collision_model = pin.buildGeomFromUrdf(
                self.model,
                str(self.urdf_path),
                pin.GeometryType.COLLISION,
                self.package_dirs,
            )
            self.collision_model.addAllCollisionPairs()
            self.collision_data = pin.GeometryData(self.collision_model)
            self._configure_collision_pairs()
            self.collision_available = self.collision_pair_count_active > 0
            self.collision_enabled = bool(self.config.use_collision and self.collision_available)
        except Exception as exc:
            self.collision_pair_mask = np.zeros((0,), dtype=np.bool_)
            self.collision_pair_count_total = 0
            self.collision_pair_count_active = 0
            self.collision_pair_count_filtered = 0
            self.collision_available = False
            self.collision_enabled = False
            self.collision_model = None
            self.collision_data = None
            if self.config.require_collision:
                raise RuntimeError(f"failed to initialize collision model: {exc}") from exc

    def _configure_collision_pairs(self) -> None:
        if self.collision_model is None or self.collision_data is None:
            self.collision_pair_mask = np.zeros((0,), dtype=np.bool_)
            self.collision_pair_count_total = 0
            self.collision_pair_count_active = 0
            self.collision_pair_count_filtered = 0
            return

        total_pairs = int(len(self.collision_model.collisionPairs))
        mask = np.ones((total_pairs,), dtype=np.bool_)
        if total_pairs <= 0:
            self.collision_pair_mask = mask
            self.collision_pair_count_total = 0
            self.collision_pair_count_active = 0
            self.collision_pair_count_filtered = 0
            return

        if self.ignored_collision_links:
            mask &= ~self._ignored_link_collision_pair_mask()
        if self.config.collision_filter_adjacent_pairs:
            mask &= ~self._adjacent_collision_pair_mask()
        if self.config.collision_filter_neutral_touching_pairs:
            mask &= ~self._neutral_touching_collision_pair_mask()
        if self.forced_collision_link_pairs:
            mask |= self._force_include_link_collision_pair_mask()

        self.collision_pair_count_total = total_pairs
        self.collision_pair_count_active = int(np.count_nonzero(mask))
        self.collision_pair_count_filtered = int(total_pairs - self.collision_pair_count_active)
        self._remove_filtered_collision_pairs(mask)

    def _remove_filtered_collision_pairs(self, active_mask: np.ndarray) -> None:
        if self.collision_model is None:
            self.collision_pair_mask = np.zeros((0,), dtype=np.bool_)
            return

        if active_mask.ndim != 1:
            raise ValueError(f"collision pair mask must be 1D, got shape {active_mask.shape}")

        pair_count = len(self.collision_model.collisionPairs)
        if active_mask.shape[0] != pair_count:
            raise ValueError(
                f"collision pair mask length mismatch: expected {pair_count}, got {active_mask.shape[0]}"
            )

        if pair_count <= 0:
            self.collision_data = pin.GeometryData(self.collision_model)
            self.collision_pair_mask = np.zeros((0,), dtype=np.bool_)
            return

        active_pairs = [
            self.collision_model.collisionPairs[pair_index]
            for pair_index, is_active in enumerate(active_mask)
            if bool(is_active)
        ]
        self.collision_model.removeAllCollisionPairs()
        for collision_pair in active_pairs:
            self.collision_model.addCollisionPair(collision_pair)

        self.collision_data = pin.GeometryData(self.collision_model)
        self.collision_pair_mask = np.ones((len(self.collision_model.collisionPairs),), dtype=np.bool_)

    def _ignored_link_collision_pair_mask(self) -> np.ndarray:
        if self.collision_model is None or not self.ignored_collision_links:
            return np.zeros((0,), dtype=np.bool_)

        mask = np.zeros((len(self.collision_model.collisionPairs),), dtype=np.bool_)
        for pair_index, pair in enumerate(self.collision_model.collisionPairs):
            first_object = self.collision_model.geometryObjects[pair.first]
            second_object = self.collision_model.geometryObjects[pair.second]
            first_link = self._geometry_link_name(first_object.name)
            second_link = self._geometry_link_name(second_object.name)
            if first_link in self.ignored_collision_links or second_link in self.ignored_collision_links:
                mask[pair_index] = True
        return mask

    def _force_include_link_collision_pair_mask(self) -> np.ndarray:
        if self.collision_model is None or not self.forced_collision_link_pairs:
            return np.zeros((0,), dtype=np.bool_)

        mask = np.zeros((len(self.collision_model.collisionPairs),), dtype=np.bool_)
        for pair_index, pair in enumerate(self.collision_model.collisionPairs):
            first_object = self.collision_model.geometryObjects[pair.first]
            second_object = self.collision_model.geometryObjects[pair.second]
            first_link = self._geometry_link_name(first_object.name)
            second_link = self._geometry_link_name(second_object.name)
            if frozenset((first_link, second_link)) in self.forced_collision_link_pairs:
                mask[pair_index] = True
        return mask

    def _adjacent_collision_pair_mask(self) -> np.ndarray:
        if self.collision_model is None:
            return np.zeros((0,), dtype=np.bool_)

        mask = np.zeros((len(self.collision_model.collisionPairs),), dtype=np.bool_)
        for pair_index, pair in enumerate(self.collision_model.collisionPairs):
            first_object = self.collision_model.geometryObjects[pair.first]
            second_object = self.collision_model.geometryObjects[pair.second]
            first_link = self._geometry_link_name(first_object.name)
            second_link = self._geometry_link_name(second_object.name)
            if (
                first_link == second_link
                or frozenset((first_link, second_link)) in self.adjacent_collision_link_pairs
            ):
                mask[pair_index] = True
        return mask

    def _neutral_touching_collision_pair_mask(self) -> np.ndarray:
        if self.collision_model is None or self.collision_data is None:
            return np.zeros((0,), dtype=np.bool_)

        try:
            self._update_collision_distances(self.neutral)
        except Exception:
            return np.zeros((len(self.collision_model.collisionPairs),), dtype=np.bool_)

        tolerance = float(self.config.collision_neutral_touching_tolerance)
        mask = np.zeros((len(self.collision_model.collisionPairs),), dtype=np.bool_)
        for pair_index, result in enumerate(self.collision_data.distanceResults):
            distance = getattr(result, "min_distance", None)
            if distance is None:
                distance = getattr(result, "minDistance", None)
            if distance is None:
                continue
            value = float(distance)
            if not np.isfinite(value):
                continue
            if value <= tolerance:
                mask[pair_index] = True
        return mask

    def _update_collision_distances(self, q: np.ndarray) -> None:
        if self.collision_model is None or self.collision_data is None:
            raise RuntimeError("collision model is not initialized")
        pin.updateGeometryPlacements(
            self.model,
            self.data,
            self.collision_model,
            self.collision_data,
            q,
        )
        try:
            pin.computeDistances(
                self.model,
                self.data,
                self.collision_model,
                self.collision_data,
                q,
            )
        except Exception:
            pin.computeDistances(self.collision_model, self.collision_data)

    def _build_adjacent_collision_link_pairs(self) -> set[frozenset[str]]:
        adjacent_pairs: set[frozenset[str]] = set()
        try:
            root = ET.parse(self.urdf_path).getroot()
        except Exception:
            return adjacent_pairs

        for joint in root.findall("joint"):
            parent = joint.find("parent")
            child = joint.find("child")
            if parent is None or child is None:
                continue
            parent_link = parent.attrib.get("link")
            child_link = child.attrib.get("link")
            if not parent_link or not child_link:
                continue
            adjacent_pairs.add(frozenset((str(parent_link), str(child_link))))
        return adjacent_pairs

    @staticmethod
    def _geometry_link_name(geometry_name: str) -> str:
        base_name = str(geometry_name)
        prefix, separator, suffix = base_name.rpartition("_")
        if separator and suffix.isdigit():
            return prefix
        return base_name

    @staticmethod
    def _normalize_link_pair_set(values: Any) -> set[frozenset[str]]:
        if not isinstance(values, (list, tuple)):
            return set()
        normalized: set[frozenset[str]] = set()
        for item in values:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            first = str(item[0]).strip()
            second = str(item[1]).strip()
            if not first or not second or first == second:
                continue
            normalized.add(frozenset((first, second)))
        return normalized

    def _clip_q_indices(
        self,
        q: np.ndarray,
        *,
        q_indices: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> np.ndarray:
        clipped = q.astype(np.float64, copy=True)
        if q_indices.size == 0:
            return clipped

        values = clipped[q_indices]
        finite_lower = np.isfinite(lower)
        finite_upper = np.isfinite(upper)
        if finite_lower.any():
            values[finite_lower] = np.maximum(values[finite_lower], lower[finite_lower])
        if finite_upper.any():
            values[finite_upper] = np.minimum(values[finite_upper], upper[finite_upper])
        clipped[q_indices] = values
        return clipped

    def _q_indices_for_joint_names(self, joint_names: list[str]) -> np.ndarray:
        indices: list[int] = []
        for name in joint_names:
            cached = self._joint_q_index_cache.get(name)
            if cached is None:
                joint_id = self._require_joint_id(name)
                if int(self.model.nqs[joint_id]) != 1 or int(self.model.nvs[joint_id]) != 1:
                    raise ValueError(
                        f"joint '{name}' is not 1-DoF (nq={self.model.nqs[joint_id]}, nv={self.model.nvs[joint_id]})."
                    )
                cached = int(self.model.idx_qs[joint_id])
                self._joint_q_index_cache[name] = cached
            indices.append(cached)
        return np.asarray(indices, dtype=np.int64)

    def _require_joint_id(self, name: str) -> int:
        joint_id = int(self.model.getJointId(name))
        if joint_id == 0:
            raise KeyError(f"joint '{name}' not found in URDF model")
        return joint_id

    def _require_frame_id(self, name: str | None) -> int:
        if not name:
            raise ValueError("frame name cannot be empty")
        frame_id = int(self.model.getFrameId(name))
        if frame_id >= len(self.model.frames):
            raise KeyError(f"frame '{name}' not found in URDF model")
        return frame_id
