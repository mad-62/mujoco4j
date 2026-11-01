package org.mujoco4j.blackbox;

import org.junit.jupiter.api.Test;
import org.mujoco4j.*;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Black-box verification of contact detection, normal contact equilibrium force,
 * and dynamic {@code d->arena} contact list views.
 */
class ContactEquilibriumBlackBoxTest {

    private static final String CONTACT_XML = """
            <mujoco model="sphere_on_ground">
              <option timestep="0.002" gravity="0 0 -9.81" solver="Newton" iterations="100" tolerance="1e-12"/>
              <worldbody>
                <geom name="ground" type="plane" size="5 5 0.1"/>
                <body name="ball" pos="0 0 0.25">
                  <freejoint name="ball_free"/>
                  <geom name="sphere_geom" type="sphere" size="0.2" mass="2.5" solref="0.01 1"/>
                </body>
              </worldbody>
            </mujoco>
            """;

    @Test
    void testSphereSettlesOnPlaneWithExactGravityEquilibriumForce() {
        try (MjModel model = MjModel.loadFromXmlString(CONTACT_XML);
             MjData data = model.makeData()) {

            double mass = 2.5;
            double g = 9.81;
            double expectedNormalForce = mass * g; // 24.525 N

            assertEquals(mass, model.totalMass(), 1e-12);

            // Initially at z = 0.25 > radius 0.20, so 0 contacts
            MjSim.forward(model, data);
            assertEquals(0, data.ncon());
            assertTrue(data.contacts().isEmpty());

            // Step 1500 steps (3.0 seconds) until the sphere settles into static equilibrium on the plane
            for (int i = 0; i < 1500; i++) {
                MjSim.step(model, data);
            }

            assertEquals(1, data.ncon());
            assertEquals(1, data.contacts().size());

            MjContact c = data.contact(0);
            assertTrue(c.dist() <= 0.0, "Contact penetration distance should be <= 0 in equilibrium");
            // Normal points upward (+z) from ground plane to sphere
            assertEquals(0.0, c.frame().get(0), 1e-9);
            assertEquals(0.0, c.frame().get(1), 1e-9);
            assertEquals(1.0, c.frame().get(2), 1e-9);

            // Generalized vertical constraint force must balance weight m * g = 24.525 N
            double fz = data.qfrcConstraint().get(2);
            assertEquals(expectedNormalForce, fz, 1e-3);

            // 6D contact wrench via mj_contactForce
            double[] wrench = MjSim.contactForce(model, data, 0);
            assertEquals(expectedNormalForce, wrench[0], 1e-3);

            // Detached snapshot survives subsequent steps
            List<MjContact> snapshot = data.contacts().copyAll();
            assertEquals(1, snapshot.size());
            assertEquals(c.dist(), snapshot.get(0).dist(), 0.0);
        }
    }
}
