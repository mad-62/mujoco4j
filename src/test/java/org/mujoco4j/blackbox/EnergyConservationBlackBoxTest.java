package org.mujoco4j.blackbox;

import org.junit.jupiter.api.Test;
import org.mujoco4j.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Black-box verification of total mechanical energy conservation (\(E = E_{\text{potential}} + E_{\text{kinetic}}\))
 * for an undamped double pendulum integrated with {@code RK4} and {@code <flag energy="enable"/>}.
 */
class EnergyConservationBlackBoxTest {

    private static final String PENDULUM_XML = """
            <mujoco model="energy_conserving_pendulum">
              <option timestep="0.001" integrator="RK4" gravity="0 0 -9.81">
                <flag energy="enable" contact="disable"/>
              </option>
              <worldbody>
                <body name="link1" pos="0 0 2">
                  <joint name="pin1" type="hinge" axis="0 1 0" damping="0" frictionloss="0"/>
                  <geom type="capsule" fromto="0 0 0 0 0 -0.8" size="0.04" mass="1.5"/>
                  <body name="link2" pos="0 0 -0.8">
                    <joint name="pin2" type="hinge" axis="0 1 0" damping="0" frictionloss="0"/>
                    <geom type="capsule" fromto="0 0 0 0 0 -0.8" size="0.04" mass="1.0"/>
                  </body>
                </body>
              </worldbody>
            </mujoco>
            """;

    @Test
    void testUndampedPendulumConservesMechanicalEnergy() {
        try (MjModel model = MjModel.loadFromXmlString(PENDULUM_XML);
             MjData data = model.makeData()) {

            assertTrue(model.opt().isEnabled(MjEnableBit.ENERGY));
            assertEquals(MjIntegrator.RK4, model.opt().integratorEnum());

            // Displace both links to a high-energy initial configuration
            data.qpos().set(0, Math.PI / 3.0);
            data.qpos().set(1, -Math.PI / 4.0);
            MjSim.forward(model, data);

            double initialPotential = data.energyPotential();
            double initialKinetic = data.energyKinetic();
            double initialTotalEnergy = data.totalEnergy();
            assertEquals(0.0, initialKinetic, 1e-14);
            assertEquals(initialPotential + initialKinetic, initialTotalEnergy, 1e-14);

            double maxDrift = 0.0;
            double maxKinetic = 0.0;
            for (int step = 0; step < 2000; step++) {
                MjSim.step(model, data);
                double currentTotal = data.totalEnergy();
                double drift = Math.abs(currentTotal - initialTotalEnergy);
                maxDrift = Math.max(maxDrift, drift);
                maxKinetic = Math.max(maxKinetic, data.energyKinetic());
            }

            // Kinetic energy must have become non-zero during swing
            assertTrue(maxKinetic > 1.0, "Max kinetic energy was " + maxKinetic);
            // Total mechanical energy drift over 2.0s with RK4 (dt = 0.001s) must be < 1e-6 J
            assertTrue(maxDrift < 1e-6, "Max mechanical energy drift was " + maxDrift);
        }
    }
}
