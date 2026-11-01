package org.mujoco4j.blackbox;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.MethodSource;
import org.mujoco4j.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Black-box verification of Google DeepMind's official C++ engine tests ({@code engine_forward_test.cc})
 * and all official MuJoCo 3.17.0 benchmark models in {@code .mujoco/model/}.
 */
class DeepMindEngineAndModelsBlackBoxTest {

    static Stream<Path> officialModelXmlFiles() throws IOException {
        Path modelDir = Path.of(".mujoco", "model");
        assertTrue(Files.isDirectory(modelDir), "Official .mujoco/model directory must exist");
        List<Path> xmlFiles;
        try (Stream<Path> walk = Files.walk(modelDir)) {
            xmlFiles = walk
                    .filter(Files::isRegularFile)
                    .filter(p -> p.getFileName().toString().endsWith(".xml"))
                    // Exclude pure include-fragment XMLs if any exist; all top-level models in .mujoco/model are full scenes
                    .sorted()
                    .toList();
        }
        assertFalse(xmlFiles.isEmpty(), "Expected official MuJoCo XML models in .mujoco/model");
        return xmlFiles.stream();
    }

    @ParameterizedTest(name = "{index}: {0}")
    @MethodSource("officialModelXmlFiles")
    void testOfficialMuJoCoModelsLoadForwardAndStep(Path xmlPath) {
        try (MjModel model = MjModel.loadFromXmlPath(xmlPath);
             MjData data = model.makeData()) {
            assertTrue(model.nbody() >= 1);
            MjSim.forward(model, data);
            MjSim.step(model, data);
            assertTrue(data.time() > 0.0, "Simulation time did not advance for " + xmlPath);

            // Verify no NaN in generalized coordinates or velocities after stepping
            for (int i = 0; i < model.nq(); i++) {
                assertFalse(Double.isNaN(data.qpos().get(i)), "NaN qpos[" + i + "] in " + xmlPath);
            }
            for (int i = 0; i < model.nv(); i++) {
                assertFalse(Double.isNaN(data.qvel().get(i)), "NaN qvel[" + i + "] in " + xmlPath);
            }
        }
    }

    @ParameterizedTest
    @EnumSource(MjIntegrator.class)
    void testActLimitedAcrossAllIntegrators(MjIntegrator integrator) {
        String xml = """
                <mujoco>
                  <option timestep="0.01"/>
                  <worldbody>
                    <body>
                      <joint name="slide" type="slide" axis="1 0 0"/>
                      <geom size=".1"/>
                    </body>
                  </worldbody>
                  <actuator>
                    <general joint="slide" gainprm="100" biasprm="0 -100" biastype="affine"
                             dynprm="10" dyntype="integrator"
                             actlimited="true" actrange="-1 1"/>
                  </actuator>
                </mujoco>
                """;
        try (MjModel model = MjModel.loadFromXmlString(xml);
             MjData data = model.makeData()) {
            model.opt().setIntegrator(integrator);

            data.ctrl().set(0, 1.0);
            for (int i = 0; i < 200; i++) {
                MjSim.step(model, data);
                assertTrue(data.act().get(0) > -1.0);
                if (i < 99) {
                    assertTrue(data.act().get(0) < 1.0);
                } else {
                    assertEquals(1.0, data.act().get(0), 5e-6);
                }
            }

            data.ctrl().set(0, -1.0);
            for (int i = 0; i < 300; i++) {
                MjSim.step(model, data);
                assertTrue(data.act().get(0) < model.actuatorActrange().get(1));
                if (i < 199) {
                    assertTrue(data.act().get(0) > model.actuatorActrange().get(0));
                } else {
                    assertEquals(model.actuatorActrange().get(0), data.act().get(0), 5e-6);
                }
            }
        }
    }

    @Test
    void testDamperDampens() {
        String xml = """
                <mujoco>
                  <worldbody>
                    <body>
                      <geom size="1"/>
                      <joint name="jnt" type="slide" axis="1 0 0"/>
                    </body>
                  </worldbody>
                  <actuator>
                    <motor joint="jnt"/>
                    <damper joint="jnt" kv="1000" ctrlrange="0 100"/>
                  </actuator>
                </mujoco>
                """;
        try (MjModel model = MjModel.loadFromXmlString(xml);
             MjData data = model.makeData()) {
            data.ctrl().set(0, 100.0);
            data.ctrl().set(1, 0.0);
            for (int i = 0; i < 100; i++) {
                MjSim.step(model, data);
            }
            assertTrue(data.qvel().get(0) > 0.004);

            data.ctrl().set(0, 0.0);
            data.ctrl().set(1, 100.0);
            for (int i = 0; i < 1000; i++) {
                MjSim.step(model, data);
            }
            assertTrue(Math.abs(data.qvel().get(0)) <= Math.ulp(1.0));
        }
    }

    @Test
    void testForwardInverseDynamicsIdentityOnHumanoid() {
        Path humanoidPath = Path.of(".mujoco", "model", "humanoid", "humanoid.xml");
        try (MjModel model = MjModel.loadFromXmlPath(humanoidPath);
             MjData data = model.makeData()) {
            // Step humanoid 25 times with control excitation so contacts and forces are active
            for (int step = 0; step < 25; step++) {
                for (int i = 0; i < model.nu(); i++) {
                    data.ctrl().set(i, 0.1 * Math.sin(step + i));
                }
                MjSim.step(model, data);
            }

            MjSim.forward(model, data);
            MjSim.inverse(model, data);
            MjSim.compareFwdInv(model, data);

            // Forward and inverse dynamics must agree to high precision
            assertTrue(data.solverFwdinv().get(0) < 1e-9, "solver_fwdinv[0] too large: " + data.solverFwdinv().get(0));
            assertTrue(data.solverFwdinv().get(1) < 1e-9, "solver_fwdinv[1] too large: " + data.solverFwdinv().get(1));

            for (int i = 0; i < model.nv(); i++) {
                double expectedInverse = data.qfrcApplied().get(i) + data.qfrcActuator().get(i);
                assertEquals(expectedInverse, data.qfrcInverse().get(i), 1e-7, "Mismatch at dof " + i);
            }
        }
    }
}
