from __future__ import annotations

import math
from typing import Any

from egga.supervisor.friction_circle import (
    FrictionCirclePolicy,
    FrictionDemand,
    allocate_friction_circle,
)
from egga.supervisor.jackknife_guard import JackknifeConfig, JackknifeInputs, step_jackknife_guard


class EggaSupervisorController:
    """Core runtime safety cage logic for ROS 2 and robotics platforms.

    Zero-dependency controller logic decoupled from ROS 2 middleware.
    """

    def __init__(self, wheelbase: float = 2.8, steer_hw_max: float = 0.60) -> None:
        self.wheelbase = wheelbase
        self.steer_hw_max = steer_hw_max
        self.mu_est = 0.8
        self.jackknife_cfg = JackknifeConfig()
        self.current_speed = 0.0

    def set_friction_estimate(self, mu: float) -> None:
        if math.isfinite(mu) and mu > 0.0:
            self.mu_est = mu

    def filter_command(
        self,
        v_req: float,
        omega_req: float,
        current_speed: float,
        theta_a: float = 0.0,
        theta_a_dot: float = 0.0,
        ltr: float = 0.0,
        dt: float = 0.01,
    ) -> dict[str, Any]:
        """Applies 2D friction circle governance and anti-jackknife protection to raw commands."""
        self.current_speed = current_speed
        v_eff = max(current_speed, 0.5)

        # Approximate equivalent steering angle and accelerations from kinematic bicycle model
        delta_req = math.atan2(omega_req * self.wheelbase, v_eff)
        delta_req = min(max(delta_req, -self.steer_hw_max), self.steer_hw_max)

        ax_req = (v_req - current_speed) / dt if dt > 0.0 else 0.0
        ay_req = (v_eff * v_eff / self.wheelbase) * math.tan(delta_req)

        # 1. 2D Friction Circle Allocation
        fc_demand = FrictionDemand(
            ax_req=ax_req,
            ay_req=ay_req,
            mu=self.mu_est,
            policy=FrictionCirclePolicy.STEERING_PRIORITY,
        )
        fc_alloc = allocate_friction_circle(
            fc_demand,
            speed=v_eff,
            wheelbase=self.wheelbase,
            steer_hw_max=self.steer_hw_max,
        )

        # Safe speed from allocated longitudinal acceleration
        v_safe = max(current_speed + fc_alloc.ax_safe * dt, 0.0)

        # 2. Articulated Jackknife & Rollover Guard
        jk_inp = JackknifeInputs(
            theta_a=theta_a,
            theta_a_dot=theta_a_dot,
            vx=v_eff,
            mu=self.mu_est,
            ltr=ltr,
            steer_cmd_req=delta_req,
            steer_rate_req=(delta_req / dt) if dt > 0.0 else 0.0,
        )
        jk_out = step_jackknife_guard(self.jackknife_cfg, jk_inp)

        # Final guarded steering angle: bounded by both friction circle and jackknife guard
        delta_safe = min(
            max(jk_out.steer_cmd_safe, -fc_alloc.delta_max_coupled),
            fc_alloc.delta_max_coupled,
        )
        omega_safe = (v_safe / self.wheelbase) * math.tan(delta_safe)

        is_emergency = jk_out.is_rollover_critical or (fc_alloc.utilisation > 1.2)

        return {
            "linear_velocity_safe": v_safe,
            "angular_velocity_safe": omega_safe,
            "steering_angle_safe": delta_safe,
            "friction_utilisation": fc_alloc.utilisation,
            "is_friction_clamped": fc_alloc.is_clamped,
            "trailer_brake_pressure": jk_out.trailer_brake_pressure,
            "is_jackknife_critical": jk_out.is_jackknife_critical,
            "is_rollover_critical": jk_out.is_rollover_critical,
            "emergency_stop": is_emergency,
        }


def main() -> None:
    """ROS 2 Node entry point (executes when rclpy environment is available)."""
    try:
        import rclpy  # type: ignore[import-not-found]
        from geometry_msgs.msg import Twist  # type: ignore[import-not-found]
        from nav_msgs.msg import Odometry  # type: ignore[import-not-found]
        from rclpy.node import Node  # type: ignore[import-not-found]
        from std_msgs.msg import Bool, Float32  # type: ignore[import-not-found]

        class EggaSupervisorNode(Node):  # type: ignore[misc]
            def __init__(self) -> None:
                super().__init__("egga_supervisor_node")
                self.controller = EggaSupervisorController()
                self.curr_vx = 0.0

                self.sub_odom = self.create_subscription(
                    Odometry, "/odom", self._on_odom, 10
                )
                self.sub_cmd = self.create_subscription(
                    Twist, "/cmd_vel_raw", self._on_cmd_vel, 10
                )
                self.sub_mu = self.create_subscription(
                    Float32, "/road_friction", self._on_friction, 10
                )

                self.pub_cmd = self.create_publisher(Twist, "/cmd_vel_safe", 10)
                self.pub_estop = self.create_publisher(Bool, "/emergency_stop", 10)
                self.get_logger().info("EGGA Runtime Supervisor Node initialized (ASIL D Ready).")

            def _on_odom(self, msg: Any) -> None:
                self.curr_vx = float(msg.twist.twist.linear.x)

            def _on_friction(self, msg: Any) -> None:
                self.controller.set_friction_estimate(float(msg.data))

            def _on_cmd_vel(self, msg: Any) -> None:
                res = self.controller.filter_command(
                    v_req=float(msg.linear.x),
                    omega_req=float(msg.angular.z),
                    current_speed=self.curr_vx,
                )
                safe_twist = Twist()
                safe_twist.linear.x = res["linear_velocity_safe"]
                safe_twist.angular.z = res["angular_velocity_safe"]
                self.pub_cmd.publish(safe_twist)

                estop = Bool()
                estop.data = res["emergency_stop"]
                self.pub_estop.publish(estop)

        rclpy.init()
        node = EggaSupervisorNode()
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()
    except ImportError:
        print(
            "[EGGA ROS 2] rclpy not detected in current environment. Standalone controller active."
        )


if __name__ == "__main__":
    main()
