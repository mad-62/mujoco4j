package org.mujoco4j.blackbox;

import org.junit.jupiter.api.Test;
import org.mujoco4j.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Black-box analytical verification of 6-DoF free-fall kinematics under uniform gravity
 * (\(g = -9.81\text{ m/s}^2\), \(\Delta t = 0.001\text{ s}\)) and quaternion normalization.
 */
class FreeFallBlackBoxTest {

    private static final String FREE_FALL_XML = """
            <mujoco model="free_fall">
              <option timestep="0.001" gravity="0 0 -9.81" integrator="RK4"/>
              <worldbody>
                <body name="ball" pos="0 0 100">
                  <freejoint name="ball_free"/>
                  <geom name="ball_geom" type="sphere" size="0.1" mass="2.0"/>
                </body>
              </worldbody>
            </mujoco>
            """;

    @Test
    void testFreeFallAnalyticalTrajectoryAndQuaternionNormalization() {
        try (MjModel model = MjModel.loadFromXmlString(FREE_FALL_XML);
             MjData data = model.makeData()) {

            // A 6-DoF free joint has 7 position coordinates (x, y, z, qw, qx, qy, qz) and 6 velocity DoFs
            assertEquals(7, model.nq());
            assertEquals(6, model.nv());

            // Initial state: z0 = 100.0 m, vx0 = 3.5 m/s, vy0 = -2.0 m/s, vz0 = 5.0 m/s
            double z0 = 100.0;
            double vx0 = 3.5;
            double vy0 = -2.0;
            double vz0 = 5.0;
            double g = -9.81;

            data.qpos().set(0, 0.0);
            data.qpos().set(1, 0.0);
            data.qpos().set(2, z0);
            // Non-unit quaternion to verify mju_normalize4 / mj_normalizeQuat
            data.qpos().set(3, 2.0);
            data.qpos().set(4, 0.0);
            data.qpos().set(5, 2.0);
            data.qpos().set(6, 0.0);
            MjSim.normalizeQuat(model, data);

            double qNorm = Math.sqrt(
                    Math.pow(data.qpos().get(3), 2) +
                    Math.pow(data.qpos().get(4), 2) +
                    Math.pow(data.qpos().get(5), 2) +
                    Math.pow(data.qpos().get(6), 2)
            );
            assertEquals(1.0, qNorm, 1e-15);

            data.qvel().set(0, vx0);
            data.qvel().set(1, vy0);
            data.qvel().set(2, vz0);

            int steps = 1000; // T = 1.0 s
            for (int i = 0; i < steps; i++) {
                MjSim.step(model, data);
            }

            double t = data.time();
            assertEquals(1.0, t, 1e-12);

            double expectedX = vx0 * t;
            double expectedY = vy0 * t;
            double expectedZ = z0 + vz0 * t + 0.5 * g * t * t;
            double expectedVz = vz0 + g * t;

            assertEquals(expectedX, data.qpos().get(0), 1e-9);
            assertEquals(expectedY, data.qpos().get(1), 1e-9);
            assertEquals(expectedZ, data.qpos().get(2), 1e-7);
            assertEquals(vx0, data.qvel().get(0), 1e-12);
            assertEquals(vy0, data.qvel().get(1), 1e-12);
            assertEquals(expectedVz, data.qvel().get(2), 1e-9);
            assertEquals(g, data.qacc().get(2), 1e-12);
        }
    }
}
