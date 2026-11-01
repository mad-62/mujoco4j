package org.mujoco4j.blackbox;

import org.junit.jupiter.api.Test;
import org.mujoco4j.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Black-box verification of bitwise-identical trajectory replay using {@link MjState#INTEGRATION}
 * (including {@code qacc_warmstart} and actuator activations {@code act}) and {@code mj_resetDataKeyframe}.
 */
class DeterminismAndStateBlackBoxTest {

    private static final String STATEFUL_XML = """
            <mujoco model="determinism_test">
              <option timestep="0.002"/>
              <worldbody>
                <geom type="plane" size="5 5 0.1"/>
                <body name="box" pos="0 0 0.15">
                  <freejoint/>
                  <geom type="box" size="0.1 0.1 0.1" mass="1.0"/>
                  <body name="arm" pos="0 0 0.1">
                    <joint name="hinge" type="hinge" axis="0 1 0"/>
                    <geom type="capsule" fromto="0 0 0 0.2 0 0.2" size="0.03" mass="0.5"/>
                  </body>
                </body>
              </worldbody>
              <actuator>
                <general name="act_hinge" joint="hinge" dyntype="filter" dynprm="0.05" gainprm="20"/>
              </actuator>
              <keyframe>
                <key name="elevated" time="0.5" qpos="0 0 0.5 1 0 0 0 0.3" act="0.25"/>
              </keyframe>
            </mujoco>
            """;

    @Test
    void testBitwiseIdenticalTrajectoryReplayWithIntegrationState() {
        try (MjModel model = MjModel.loadFromXmlString(STATEFUL_XML);
             MjData data = model.makeData()) {

            // Step 50 times to build up contacts, warmstart accelerations, and filter activations
            for (int i = 0; i < 50; i++) {
                data.ctrl().set(0, Math.sin(0.1 * i));
                MjSim.step(model, data);
            }

            // Capture full integration state (time, qpos, qvel, act, qacc_warmstart, etc.)
            double[] savedState = MjSim.getState(model, data, MjState.INTEGRATION);
            assertEquals(MjSim.stateSize(model, MjState.INTEGRATION), savedState.length);

            // Roll forward 100 steps and record final state
            for (int i = 0; i < 100; i++) {
                data.ctrl().set(0, Math.cos(0.05 * i));
                MjSim.step(model, data);
            }
            double expectedTime = data.time();
            double[] expectedQpos = data.qpos().toDoubleArray();
            double[] expectedQvel = data.qvel().toDoubleArray();
            double[] expectedQacc = data.qacc().toDoubleArray();
            double[] expectedAct = data.act().toDoubleArray();

            // Scramble simulation state completely via resetDataKeyframe, then restore savedState
            MjSim.resetDataKeyframe(model, data, 0);
            assertEquals(0.5, data.time(), 0.0);

            MjSim.setState(model, data, savedState, MjState.INTEGRATION);

            // Replay the exact same 100 control inputs
            for (int i = 0; i < 100; i++) {
                data.ctrl().set(0, Math.cos(0.05 * i));
                MjSim.step(model, data);
            }

            // Must match to 0.0 ULPs (bitwise identical)
            assertEquals(expectedTime, data.time(), 0.0);
            assertArrayEquals(expectedQpos, data.qpos().toDoubleArray(), 0.0);
            assertArrayEquals(expectedQvel, data.qvel().toDoubleArray(), 0.0);
            assertArrayEquals(expectedQacc, data.qacc().toDoubleArray(), 0.0);
            assertArrayEquals(expectedAct, data.act().toDoubleArray(), 0.0);
        }
    }
}
