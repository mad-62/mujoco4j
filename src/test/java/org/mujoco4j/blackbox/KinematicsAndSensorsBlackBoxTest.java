package org.mujoco4j.blackbox;

import org.junit.jupiter.api.Test;
import org.mujoco4j.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Black-box verification of forward kinematics (\(x = L_1 \cos\theta_1 + L_2 \cos(\theta_1 + \theta_2)\),
 * \(y = L_1 \sin\theta_1 + L_2 \sin(\theta_1 + \theta_2)\)) and proper acceleration sensor readings.
 */
class KinematicsAndSensorsBlackBoxTest {

    private static final String TWO_LINK_ARM_XML = """
            <mujoco model="planar_2link_arm">
              <option timestep="0.001" gravity="0 0 -9.81"/>
              <worldbody>
                <body name="link1" pos="0 0 0">
                  <joint name="theta1" type="hinge" axis="0 0 1"/>
                  <geom type="capsule" fromto="0 0 0 1.0 0 0" size="0.04" mass="1.0"/>
                  <body name="link2" pos="1.0 0 0">
                    <joint name="theta2" type="hinge" axis="0 0 1"/>
                    <geom type="capsule" fromto="0 0 0 0.75 0 0" size="0.03" mass="1.0"/>
                    <site name="end_effector" pos="0.75 0 0" size="0.02"/>
                  </body>
                </body>
              </worldbody>
              <sensor>
                <framepos name="ee_pos" objtype="site" objname="end_effector"/>
                <accelerometer name="ee_accel" site="end_effector"/>
              </sensor>
            </mujoco>
            """;

    @Test
    void testTwoLinkPlanarArmForwardKinematicsAndProperAcceleration() {
        try (MjModel model = MjModel.loadFromXmlString(TWO_LINK_ARM_XML);
             MjData data = model.makeData()) {

            double l1 = 1.0;
            double l2 = 0.75;
            double theta1 = Math.PI / 6.0; // 30 deg
            double theta2 = Math.PI / 3.0; // 60 deg -> theta1 + theta2 = 90 deg

            data.joint("theta1").qpos().set(0, theta1);
            data.joint("theta2").qpos().set(0, theta2);
            MjSim.forward(model, data);

            double expectedX = l1 * Math.cos(theta1) + l2 * Math.cos(theta1 + theta2);
            double expectedY = l1 * Math.sin(theta1) + l2 * Math.sin(theta1 + theta2);

            MjDoubleVector siteXpos = data.site("end_effector").xpos();
            assertEquals(expectedX, siteXpos.get(0), 1e-12);
            assertEquals(expectedY, siteXpos.get(1), 1e-12);
            assertEquals(0.0, siteXpos.get(2), 1e-12);

            // Sensor framepos matches site_xpos
            MjDoubleVector eePosSensor = data.sensor("ee_pos").data();
            assertEquals(expectedX, eePosSensor.get(0), 1e-12);
            assertEquals(expectedY, eePosSensor.get(1), 1e-12);
            assertEquals(0.0, eePosSensor.get(2), 1e-12);

            // Because motion is strictly in the horizontal XY plane (z_ddot = 0) while gravity is -9.81 m/s^2 in Z,
            // an accelerometer at the end-effector measures proper upward acceleration +9.81 m/s^2 along Z.
            MjDoubleVector eeAccelSensor = data.sensor("ee_accel").data();
            assertEquals(0.0, eeAccelSensor.get(0), 1e-9);
            assertEquals(0.0, eeAccelSensor.get(1), 1e-9);
            assertEquals(9.81, eeAccelSensor.get(2), 1e-9);
        }
    }
}
