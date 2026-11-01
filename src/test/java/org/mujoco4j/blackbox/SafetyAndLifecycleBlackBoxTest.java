package org.mujoco4j.blackbox;

import org.junit.jupiter.api.Test;
import org.mujoco4j.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Black-box verification of memory safety, error handling on malformed XML, index bounds checking,
 * use-after-close protection (on {@link MjModel}, {@link MjData}, and cached views),
 * parent-closed-before-child protection, and idempotent double-{@code close()}.
 */
class SafetyAndLifecycleBlackBoxTest {

    private static final String SIMPLE_XML = """
            <mujoco>
              <worldbody>
                <body name="box" pos="0 0 1">
                  <freejoint name="free"/>
                  <geom name="geom1" type="sphere" size="0.1"/>
                </body>
              </worldbody>
            </mujoco>
            """;

    @Test
    void testMalformedXmlThrowsDescriptiveMjException() {
        String malformedXml = "<mujoco><worldbody><geom type='not_a_valid_geom_type'/></worldbody></mujoco>";
        MjException ex = assertThrows(MjException.class, () -> MjModel.loadFromXmlString(malformedXml));
        assertNotNull(ex.getMessage());
        assertFalse(ex.getMessage().isBlank());
    }

    @Test
    void testVectorBoundsCheckingPreventsOutOfBoundsNativeAccess() {
        try (MjModel model = MjModel.loadFromXmlString(SIMPLE_XML);
             MjData data = model.makeData()) {

            MjDoubleVector qpos = data.qpos();
            assertEquals(7, qpos.size());

            assertThrows(IndexOutOfBoundsException.class, () -> qpos.get(-1));
            assertThrows(IndexOutOfBoundsException.class, () -> qpos.get(7));
            assertThrows(IndexOutOfBoundsException.class, () -> qpos.set(-1, 0.0));
            assertThrows(IndexOutOfBoundsException.class, () -> qpos.set(7, 0.0));
            assertThrows(IndexOutOfBoundsException.class, () -> qpos.slice(5, 3));
            assertThrows(IllegalArgumentException.class, () -> qpos.copyFrom(new double[2]));
        }
    }

    @Test
    void testUseAfterCloseOnModelDataAndCachedViewsThrowsIllegalStateException() {
        MjModel model = MjModel.loadFromXmlString(SIMPLE_XML);
        MjData data = model.makeData();
        MjDoubleVector cachedQpos = data.qpos();
        MjDoubleVector cachedBodyPos = model.bodyPos();
        MjOption cachedOpt = model.opt();

        // Idempotent double-close on MjData
        data.close();
        assertFalse(data.isOpen());
        assertDoesNotThrow(data::close);

        // Accessing closed MjData or its cached vector view throws IllegalStateException
        assertThrows(IllegalStateException.class, data::time);
        assertThrows(IllegalStateException.class, () -> cachedQpos.get(0));
        assertThrows(IllegalStateException.class, () -> MjSim.step(model, data));

        // Idempotent double-close on MjModel
        model.close();
        assertFalse(model.isOpen());
        assertDoesNotThrow(model::close);

        // Accessing closed MjModel or its cached views throws IllegalStateException
        assertThrows(IllegalStateException.class, model::nq);
        assertThrows(IllegalStateException.class, () -> cachedBodyPos.get(0));
        assertThrows(IllegalStateException.class, cachedOpt::timestep);
    }

    @Test
    void testClosingParentModelInvalidatesLiveChildDataAndViews() {
        MjModel model = MjModel.loadFromXmlString(SIMPLE_XML);
        MjData data = model.makeData();
        MjDoubleVector cachedQpos = data.qpos();

        // Close parent MjModel BEFORE child MjData
        model.close();
        assertFalse(model.isOpen());
        assertFalse(data.isOpen(), "Child MjData must report !isOpen() when parent MjModel is closed");

        assertThrows(IllegalStateException.class, data::time);
        assertThrows(IllegalStateException.class, () -> cachedQpos.get(0));
        assertThrows(IllegalStateException.class, () -> MjSim.forward(model, data));

        // Closing child MjData afterwards is still safe and idempotent
        assertDoesNotThrow(data::close);
    }
}
