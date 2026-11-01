package org.mujoco4j.blackbox;

import org.junit.jupiter.api.Test;
import org.mujoco4j.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Black-box verification of direct torque motor actuation (\(\alpha = \tau / I\))
 * and closed-loop PD position servo convergence.
 */
class ActuatorControlBlackBoxTest {

    private static final String MOTOR_XML = """
            <mujoco model="direct_motor">
              <option timestep="0.001" gravity="0 0 0"/>
              <worldbody>
                <body name="rotor" pos="0 0 0">
                  <inertial pos="0 0 0" mass="1.0" diaginertia="0.5 0.5 0.5"/>
                  <joint name="hinge_z" type="hinge" axis="0 0 1" damping="0" armature="0"/>
                  <geom type="cylinder" size="0.1 0.05"/>
                </body>
              </worldbody>
              <actuator>
                <motor name="torque_motor" joint="hinge_z" gear="2.0"/>
              </actuator>
            </mujoco>
            """;

    private static final String POSITION_SERVO_XML = """
            <mujoco model="position_servo">
              <option timestep="0.002" gravity="0 0 0"/>
              <worldbody>
                <body name="arm" pos="0 0 0">
                  <inertial pos="0 0 0" mass="1.0" diaginertia="0.2 0.2 0.2"/>
                  <joint name="joint1" type="hinge" axis="0 0 1" damping="2.0"/>
                  <geom type="capsule" fromto="0 0 0 0.5 0 0" size="0.04"/>
                </body>
              </worldbody>
              <actuator>
                <position name="pos_servo" joint="joint1" kp="50.0"/>
              </actuator>
            </mujoco>
            """;

    @Test
    void testDirectTorqueMotorAngularAcceleration() {
        try (MjModel model = MjModel.loadFromXmlString(MOTOR_XML);
             MjData data = model.makeData()) {

            // Moment of inertia I_z = 0.5 kg*m^2, gear = 2.0, ctrl = 1.5 -> tau = 3.0 N*m -> alpha = 6.0 rad/s^2
            data.ctrl().set(0, 1.5);
            MjSim.forward(model, data);

            assertEquals(3.0, data.qfrcActuator().get(0), 1e-12);
            assertEquals(6.0, data.qacc().get(0), 1e-12);
        }
    }

    @Test
    void testPositionServoConvergesToSetpoint() {
        try (MjModel model = MjModel.loadFromXmlString(POSITION_SERVO_XML);
             MjData data = model.makeData()) {

            double targetAngle = 0.85;
            data.ctrl().set(0, targetAngle);

            for (int i = 0; i < 1500; i++) {
                MjSim.step(model, data);
            }

            assertEquals(targetAngle, data.qpos().get(0), 1e-4);
            assertEquals(0.0, data.qvel().get(0), 1e-4);
        }
    }
}
