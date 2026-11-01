package org.mujoco4j.blackbox;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.mujoco4j.*;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Direct Java 25 black-box port of Google DeepMind's official {@code python/mujoco/bindings_test.py}
 * test suite exercising {@code org.mujoco4j.*} public APIs.
 */
class DeepMindBindingsBlackBoxTest {

    private static final String TEST_XML = """
            <mujoco model="MyTestModel">
              <compiler inertiafromgeom="true"/>
              <default>
                <joint limited="false" armature="1"/>
                <geom condim="3" material="mat"/>
                <default class="mydefault">
                  <geom condim="1"/>
                </default>
              </default>
              <option timestep="0.002" integrator="Euler"/>
              <visual>
                <quality shadowsize="2048"/>
                <headlight active="1"/>
              </visual>
              <asset>
                <texture name="tex" type="2d" builtin="checker" width="64" height="64"/>
                <material name="mat" texture="tex" texrepeat="1 1" texuniform="true" reflectance="0.5"/>
              </asset>
              <worldbody>
                <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
                <geom name="floor" type="plane" size="1 1 0.1" pos="0 0 0"/>
                <body name="box" pos="0 0 0.2">
                  <joint name="free" type="free"/>
                  <geom name="mygeom" type="box" size="0.1 0.2 0.3" mass="1"/>
                  <site name="mysite" pos="0.1 0.1 0.1" size="0.01"/>
                  <camera name="mycam" pos="0 0 0.5"/>
                  <body name="child" pos="0 0 0.3">
                    <joint name="hinge" type="hinge" axis="0 1 0" damping="0.1" stiffness="1 2 3"/>
                    <geom name="childgeom" type="sphere" size="0.1" pos="42 0 42"/>
                    <site name="mysite2" pos="0 0 0.1" size="0.01"/>
                  </body>
                </body>
                <body name="mocap_body" mocap="true" pos="1 2 3">
                  <geom name="mocap_geom" type="sphere" size="0.05"/>
                </body>
              </worldbody>
              <tendon>
                <spatial name="mytendon">
                  <site site="mysite"/>
                  <site site="mysite2"/>
                </spatial>
              </tendon>
              <actuator>
                <motor name="myactuator" joint="hinge" gear="10" ctrlrange="-1 1" ctrllimited="true"/>
              </actuator>
              <sensor>
                <jointpos name="mysensor" joint="hinge"/>
              </sensor>
              <custom>
                <numeric name="mynumeric" size="2" data="1.5 2.5"/>
                <text name="mytext" data="hello_mujoco"/>
              </custom>
              <keyframe>
                <key name="mykey" time="1.0"/>
              </keyframe>
            </mujoco>
            """;

    private static final String TEST_XML_SENSOR = """
            <mujoco>
              <worldbody>
                <body>
                  <joint name="hinge" type="hinge"/>
                  <geom size="0.1"/>
                </body>
              </worldbody>
              <sensor>
                <user name="usersensor" dim="1" needstage="vel"/>
              </sensor>
            </mujoco>
            """;

    private MjModel model;
    private MjData data;

    @BeforeEach
    void setUp() {
        MjCallbacks.clearAll();
        model = MjModel.loadFromXmlString(TEST_XML);
        data = model.makeData();
    }

    @AfterEach
    void tearDown() {
        MjCallbacks.clearAll();
        if (data != null && data.isOpen()) {
            data.close();
        }
        if (model != null && model.isOpen()) {
            model.close();
        }
    }

    @Test
    void testVersionAndConstants() {
        assertEquals(3017000, MjSim.version());
        assertEquals(MjConstants.mjVERSION_HEADER, MjSim.version());
        assertTrue(MjSim.versionString().startsWith("3.17.0"));
        assertEquals(1e-15, MjConstants.MINVAL, 1e-25);
        assertEquals(1e10, MjConstants.MAXVAL, 1.0);
        assertEquals(Math.PI, MjConstants.PI, 1e-12);
        assertEquals(2, MjConstants.NREF);
        assertEquals(5, MjConstants.NIMP);
        assertEquals(MjConstants.NDISABLE, MjConstants.disableStrings().size());
        assertEquals(MjConstants.NENABLE, MjConstants.enableStrings().size());
        assertEquals(MjConstants.NTIMER, MjConstants.timerStrings().size());
        assertEquals(MjConstants.NLABEL, MjConstants.labelStrings().size());
        assertEquals(MjConstants.NFRAME, MjConstants.frameStrings().size());
        assertEquals(MjConstants.NVISFLAG, MjConstants.visStrings().size());
        assertEquals(MjConstants.NRNDFLAG, MjConstants.rndStrings().size());
    }

    @Test
    void testCanLoadXmlAndReadNameStrings() {
        assertEquals("MyTestModel", model.modelName());
        assertEquals("box", model.body(1).name());
        assertEquals("mygeom", model.geom("mygeom").name());
        assertEquals(1, model.geom("mygeom").id());
        assertEquals("hello_mujoco", model.textDataString(0));
    }

    @Test
    void testSaveAndLoadBinaryModel(@TempDir Path tempDir) {
        Path mjbPath = tempDir.resolve("saved.mjb");
        model.saveModel(mjbPath);
        assertTrue(Files.exists(mjbPath));

        try (MjModel loaded = MjModel.loadFromBinaryPath(mjbPath)) {
            assertEquals(model.nq(), loaded.nq());
            assertEquals(model.nv(), loaded.nv());
            assertEquals(model.nbody(), loaded.nbody());
            assertEquals(model.bodyPos(), loaded.bodyPos());
        }

        byte[] bytes = model.saveModelToBuffer();
        assertEquals(model.sizeModel(), bytes.length);
        try (MjModel fromBuf = MjModel.loadFromBinaryBuffer(bytes)) {
            assertEquals(model.nq(), fromBuf.nq());
            assertEquals(model.bodyMass(), fromBuf.bodyMass());
        }
    }

    @Test
    void testVfsLoadXmlAndBinary() {
        try (MjVfs vfs = new MjVfs()) {
            vfs.addBuffer("virtual.xml", TEST_XML.getBytes(StandardCharsets.UTF_8));
            try (MjModel vfsModel = MjModel.loadFromXmlPath(Path.of("virtual.xml"), vfs)) {
                assertEquals(model.nq(), vfsModel.nq());
                assertEquals(model.nv(), vfsModel.nv());
            }

            byte[] mjb = model.saveModelToBuffer();
            vfs.addBuffer("virtual.mjb", mjb);
            try (MjModel vfsMjb = MjModel.loadFromBinaryPath(Path.of("virtual.mjb"), vfs)) {
                assertEquals(model.nq(), vfsMjb.nq());
            }

            vfs.deleteFile("virtual.xml");
            assertThrows(MjException.class, () -> vfs.deleteFile("virtual.xml"));
        }
    }

    @Test
    void testCopyModelAndData() {
        MjSim.step(model, data);
        try (MjModel modelCopy = model.copy();
             MjData dataCopy = data.copy()) {
            assertNotEquals(model.address(), modelCopy.address());
            assertNotEquals(data.address(), dataCopy.address());
            assertEquals(model.bodyPos(), modelCopy.bodyPos());
            assertEquals(data.qpos(), dataCopy.qpos());
            assertEquals(data.qvel(), dataCopy.qvel());
            assertEquals(data.time(), dataCopy.time(), 0.0);

            for (int i = 0; i < 10; i++) {
                MjSim.step(model, data);
                MjSim.step(modelCopy, dataCopy);
            }
            assertEquals(data.qpos(), dataCopy.qpos());
            assertEquals(data.qvel(), dataCopy.qvel());
            assertEquals(data.qacc(), dataCopy.qacc());
        }
    }

    @Test
    void testStructsComparisonAndCopy() {
        MjOption optCopy = model.opt().copy();
        assertEquals(model.opt(), optCopy);
        optCopy.setTimestep(0.123);
        assertNotEquals(model.opt(), optCopy);
        assertThrows(UnsupportedOperationException.class, () -> model.opt().hashCode());

        MjStatistic statCopy = model.stat().copy();
        assertEquals(model.stat(), statCopy);
        statCopy.setMeansize(99.0);
        assertNotEquals(model.stat(), statCopy);
        assertThrows(UnsupportedOperationException.class, () -> model.stat().hashCode());

        MjVisual visCopy = model.vis().copy();
        assertEquals(model.vis(), visCopy);
        visCopy.quality().setShadowsize(512);
        assertNotEquals(model.vis(), visCopy);
        assertThrows(UnsupportedOperationException.class, () -> model.vis().hashCode());
    }

    @Test
    void testNamedIndexingAndMemorySharing() {
        assertEquals(1, model.body("box").id());
        assertEquals("box", model.body(1).name());
        assertTrue(model.body("box").pos().sharesMemoryWith(model.bodyPos()));
        assertTrue(model.geom("mygeom").size().sharesMemoryWith(model.geomSize()));
        assertTrue(model.joint("hinge").stiffness().sharesMemoryWith(model.jntStiffness()));
        assertTrue(model.joint("hinge").stiffnesspoly().sharesMemoryWith(model.jntStiffnesspoly()));
        assertTrue(model.actuator("myactuator").gear().sharesMemoryWith(model.actuatorGear()));

        MjSim.forward(model, data);
        assertTrue(data.body("box").xpos().sharesMemoryWith(data.xpos()));
        assertTrue(data.geom("mygeom").xpos().sharesMemoryWith(data.geomXpos()));
        assertTrue(data.joint("free").qpos().sharesMemoryWith(data.qpos()));
        assertEquals(7, data.joint("free").qpos().size());
        assertEquals(6, data.joint("free").qvel().size());
        assertEquals(1, data.joint("hinge").qpos().size());
        assertEquals(1, data.joint("hinge").qvel().size());

        // Mutating through named indexer mutates the underlying MjData array in-place
        data.joint("hinge").qpos().set(0, 0.75);
        assertEquals(0.75, data.qpos().get(7), 0.0);

        assertThrows(IllegalArgumentException.class, () -> model.geom("nonexistent_geom"));
        assertThrows(IndexOutOfBoundsException.class, () -> model.geom(-1));
        assertThrows(IndexOutOfBoundsException.class, () -> model.geom(model.ngeom()));
    }

    @Test
    void testIndexerUnnamedEntities() {
        String xml = """
                <mujoco>
                  <worldbody>
                    <geom name="mygeom" size="1" pos="0 0 1"/>
                    <geom size="2" pos="0 0 2"/>
                    <geom size="3" pos="0 0 3"/>
                    <geom name="myothergeom" size="4" pos="0 0 4"/>
                    <geom size="5" pos="0 0 5"/>
                  </worldbody>
                </mujoco>
                """;
        try (MjModel m = MjModel.loadFromXmlString(xml);
             MjData d = m.makeData()) {
            assertEquals(0, m.geom("mygeom").id());
            assertEquals(3, m.geom("myothergeom").id());
            assertEquals("mygeom", m.geom(0).name());
            assertEquals("", m.geom(1).name());
            assertEquals("", m.geom(2).name());
            assertEquals("myothergeom", m.geom(3).name());
            assertEquals("", m.geom(4).name());
            assertEquals(1.0, m.geom(0).size().get(0), 0.0);
            assertEquals(2.0, m.geom(1).size().get(0), 0.0);
            assertEquals(3.0, m.geom(2).size().get(0), 0.0);
            assertEquals(4.0, m.geom(3).size().get(0), 0.0);
            assertEquals(5.0, m.geom(4).size().get(0), 0.0);

            MjSim.forward(m, d);
            assertEquals(1.0, d.geom(0).xpos().get(2), 0.0);
            assertEquals(2.0, d.geom(1).xpos().get(2), 0.0);
            assertEquals(3.0, d.geom(2).xpos().get(2), 0.0);
            assertEquals(4.0, d.geom(3).xpos().get(2), 0.0);
            assertEquals(5.0, d.geom(4).xpos().get(2), 0.0);
        }
    }

    @Test
    void testForwardStepAndMassMatrix() {
        MjSim.forward(model, data);
        double[] denseM = MjSim.fullM(model, data);
        assertEquals(model.nv() * model.nv(), denseM.length);

        double[] vec = new double[model.nv()];
        Arrays.fill(vec, 1.0);
        double[] resSparse = new double[model.nv()];
        MjSim.mulM(model, data, resSparse, vec);

        double[] resDense = new double[model.nv()];
        MjMath.mulMatVec(resDense, denseM, vec, model.nv(), model.nv());
        assertArrayEquals(resDense, resSparse, 1e-11);
    }

    @Test
    void testJacobians() {
        MjSim.forward(model, data);
        int nv = model.nv();
        double[] jacp = new double[3 * nv];
        double[] jacr = new double[3 * nv];
        MjSim.jacBody(model, data, jacp, jacr, model.body("box").id());

        double[] jacGeomP = new double[3 * nv];
        double[] jacGeomR = new double[3 * nv];
        MjSim.jacGeom(model, data, jacGeomP, jacGeomR, model.geom("mygeom").id());
        assertArrayEquals(jacp, jacGeomP, 1e-12);
        assertArrayEquals(jacr, jacGeomR, 1e-12);
    }

    @Test
    void testCallbacksTimeControlSensorWarning() {
        // 1. Custom time callback and restore default timer
        AtomicInteger timeCalls = new AtomicInteger();
        MjCallbacks.setTimeCallback(() -> {
            timeCalls.incrementAndGet();
            return 0.0;
        });
        MjSim.step(model, data);
        assertTrue(timeCalls.get() > 0);
        assertEquals(0.0, data.timer(MjTimer.STEP).duration(), 0.0);

        MjCallbacks.setTimeCallback(null);
        assertNull(MjCallbacks.getTimeCallback());
        int countAfterRestore = timeCalls.get();
        MjSim.step(model, data);
        assertEquals(countAfterRestore, timeCalls.get());
        assertTrue(data.timer(MjTimer.STEP).duration() > 0.0);

        // 2. Control callback
        AtomicInteger ctrlCalls = new AtomicInteger();
        MjCallbacks.setControlCallback((m, d) -> {
            assertSame(model, m);
            assertSame(data, d);
            d.ctrl().set(0, 0.5);
            ctrlCalls.incrementAndGet();
        });
        MjSim.step(model, data);
        assertEquals(1, ctrlCalls.get());
        assertEquals(0.5, data.ctrl().get(0), 0.0);
        MjCallbacks.setControlCallback(null);

        // 3. Sensor callback
        try (MjModel sensorModel = MjModel.loadFromXmlString(TEST_XML_SENSOR);
             MjData sensorData = sensorModel.makeData()) {
            AtomicInteger sensorCalls = new AtomicInteger();
            MjCallbacks.setSensorCallback((m, d, stage) -> {
                assertSame(sensorModel, m);
                assertSame(sensorData, d);
                assertEquals(MjStage.VEL, stage);
                d.sensordata().set(0, 17.0);
                sensorCalls.incrementAndGet();
            });
            MjSim.forward(sensorModel, sensorData);
            assertEquals(1, sensorCalls.get());
            assertEquals(17.0, sensorData.sensordata().get(0), 0.0);
            MjCallbacks.setSensorCallback(null);
        }

        // 4. Exception thrown inside callback is trapped and rethrown cleanly in Java
        MjCallbacks.setControlCallback((m, d) -> {
            throw new IllegalArgumentException("Expected test callback error");
        });
        IllegalArgumentException thrown = assertThrows(
                IllegalArgumentException.class,
                () -> MjSim.step(model, data)
        );
        assertEquals("Expected test callback error", thrown.getMessage());
        MjCallbacks.setControlCallback(null);

        // 5. User warning callback
        List<String> warnings = new ArrayList<>();
        MjCallbacks.setWarningCallback(warnings::add);
        data.qpos().set(0, Double.NaN);
        MjSim.checkPos(model, data);
        assertFalse(warnings.isEmpty());
        assertEquals(1, data.warning(MjWarning.BADQPOS).number());
        MjCallbacks.setWarningCallback(null);
        MjSim.resetData(model, data);
    }

    @Test
    void testRayAndMultiRayAndGeomDistance() {
        MjSim.forward(model, data);
        int[] geomid = new int[1];
        double dist = MjRay.ray(
                model, data,
                new double[]{0, 0, 0},
                new double[]{0, 0, 1},
                null, true, -1, geomid, null
        );
        assertTrue(dist >= -1.0);

        // mju_rayGeom with a plane at origin
        double[] pos = {0.0, 0.0, 0.0};
        double[] mat = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        double[] size = {10.0, 10.0, 1.0};
        double[] pnt = {5.0, 5.0, 5.0};
        double invNorm = 1.0 / Math.sqrt(3.0);
        double[] vec = {-invNorm, -invNorm, -invNorm};
        double[] normal = new double[3];
        double rayGeomDist = MjRay.rayGeom(pos, mat, size, pnt, vec, MjGeomType.PLANE.value(), normal);
        double expectedDist = Math.sqrt(3.0 * 25.0);
        assertEquals(expectedDist, rayGeomDist, 1e-9);
        assertArrayEquals(new double[]{0.0, 0.0, 1.0}, normal, 1e-9);

        // mj_multiRay
        int nray = 3;
        double[] multiPnt = {-0.3, 0.0, 0.1};
        double[] multiVec = {
                1.0, 0.0, 0.0,
                0.0, 0.0, 1.0,
                0.0, 0.0, -1.0
        };
        int[] multiGeomId = new int[nray];
        double[] multiDist = new double[nray];
        MjRay.multiRay(
                model, data, multiPnt, multiVec, null, true, -1,
                multiGeomId, multiDist, null, nray, MjConstants.MAXVAL
        );
        double[] expectedDists = {0.2, -1.0, 0.1};
        int[] expectedGeoms = {1, -1, 0};
        for (int i = 0; i < nray; i++) {
            int[] singleGeom = new int[1];
            double[] vecI = Arrays.copyOfRange(multiVec, 3 * i, 3 * i + 3);
            double singleDist = MjRay.ray(model, data, multiPnt, vecI, null, true, -1, singleGeom, null);
            assertEquals(singleDist, multiDist[i], 1e-12);
            assertEquals(singleGeom[0], multiGeomId[i]);
            assertEquals(expectedGeoms[i], multiGeomId[i]);
            assertEquals(expectedDists[i], multiDist[i], 1e-7);
        }

        // mj_geomDistance
        double[] fromto = new double[6];
        double gdist = MjSim.geomDistance(model, data, 0, 2, 200.0, fromto);
        assertEquals(42.4, gdist, 1e-4);
    }

    @Test
    void testInverseFdAndDerivatives() {
        double eps = 1e-6;
        MjSim.inverseFD(model, data, eps, false, null, null, null, null, null, null, null);

        int nv = model.nv();
        int ns = model.nsensordata();
        int nC = model.nC();
        double[] dfDq = new double[nv * nv];
        double[] dfDv = new double[nv * nv];
        double[] dfDa = new double[nv * nv];
        double[] dsDq = new double[nv * ns];
        double[] dsDv = new double[nv * ns];
        double[] dsDa = new double[nv * ns];
        double[] dmDq = new double[nv * nC];
        MjSim.inverseFD(model, data, eps, false, dfDq, dfDv, dfDa, dsDq, dsDv, dsDa, dmDq);
        assertTrue(MjMath.norm(dfDq) > eps);
        assertTrue(MjMath.norm(dfDv) > eps);
        assertTrue(MjMath.norm(dfDa) > eps);

        double[] q1 = {0.2, 0.3, 0.3, 0.4};
        double[] q2 = {0.1, 0.2, 0.4, 0.5};
        double[] d1 = new double[9];
        double[] d2 = new double[9];
        double[] d3 = new double[9];
        double[] d4 = new double[9];
        MjMath.subQuat(q1, q2, d1, d2);
        MjMath.subQuat(q1, q2, null, d3);
        MjMath.subQuat(q1, q2, d4, null);
        assertArrayEquals(d2, d3, 0.0);
        assertArrayEquals(d1, d4, 0.0);
    }

    @Test
    void testMathUtilitiesAndBandedAndBoxQp() {
        // mju_fill, mju_eye, mju_symmetrize, mju_clip, mju_mulVecMatVec
        double[] fillArr = new double[3];
        MjMath.fill(fillArr, 1.5);
        assertArrayEquals(new double[]{1.5, 1.5, 1.5}, fillArr, 0.0);

        double[] eye4 = new double[16];
        MjMath.eye(eye4, 4);
        assertEquals(1.0, eye4[0], 0.0);
        assertEquals(1.0, eye4[5], 0.0);
        assertEquals(1.0, eye4[10], 0.0);
        assertEquals(1.0, eye4[15], 0.0);
        assertEquals(0.0, eye4[1], 0.0);

        assertEquals(1.5, MjMath.clip(1.5, 1.0, 2.0), 0.0);
        assertEquals(2.0, MjMath.clip(1.5, 2.0, 3.0), 0.0);
        assertEquals(1.0, MjMath.clip(1.5, 0.0, 1.0), 0.0);

        double[] v1 = {1.0, 2.0, 3.0};
        double[] v2 = {3.0, 2.0, 1.0};
        double[] mat = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        assertEquals(204.0, MjMath.mulVecMatVec(v1, mat, v2, 3), 1e-12);

        // mju_euler2Quat
        double[] quat = new double[4];
        MjMath.euler2Quat(quat, new double[]{0.0, Math.PI / 2.0, 0.0}, "xyz");
        assertArrayEquals(new double[]{Math.sqrt(0.5), 0.0, Math.sqrt(0.5), 0.0}, quat, 1e-5);

        // Banded matrices
        int nTotal = 4;
        int nBand = 1;
        int nDense = 1;
        double[] dense = {
                1.0, 0.0, 0.0, 0.1,
                0.0, 2.0, 0.0, 0.2,
                0.0, 0.0, 3.0, 0.3,
                0.1, 0.2, 0.3, 4.0
        };
        double[] band = new double[nBand * (nTotal - nDense) + nDense * nTotal];
        MjMath.dense2Band(band, dense, nTotal, nBand, nDense);
        for (int i = 0; i < 4; i++) {
            int idx = MjMath.bandDiag(i, nTotal, nBand, nDense);
            assertEquals(i + 1.0, band[idx], 0.0);
        }
        double[] dense2 = new double[nTotal * nTotal];
        MjMath.band2Dense(dense2, band, nTotal, nBand, nDense, true);
        assertArrayEquals(dense, dense2, 0.0);

        // BoxQP
        int n = 5;
        double[] res = new double[n];
        double[] r = new double[n * (n + 7)];
        int[] index = new int[n];
        double[] h = new double[n * n];
        MjMath.eye(h, n);
        double[] g = new double[n];
        double[] lower = new double[n];
        double[] upper = new double[n];
        Arrays.fill(g, 1.0);
        Arrays.fill(lower, -1.0);
        Arrays.fill(upper, 1.0);
        int rank = MjMath.boxQP(res, r, index, h, g, n, lower, upper);
        assertTrue(rank > -1);
    }

    @ParameterizedTest
    @CsvSource({"false,false", "false,true", "true,false", "true,true"})
    void testPrintSchema(boolean flgHtml, boolean flgPad) {
        String schema = MjSim.printSchema(flgHtml, flgPad);
        assertTrue(schema.contains("mujoco"));
    }

    @Test
    void testPolyStiffness() {
        double x = 0.5;
        data.joint("hinge").qpos().set(0, x);
        MjSim.forward(model, data);
        double expected = -(1.0 * x + 2.0 * x * Math.abs(x) + 3.0 * Math.pow(x, 3));
        assertEquals(expected, data.joint("hinge").qfrcSpring().get(0), 1e-12);
    }

    @Test
    void testInsideSiteSphereAndBox() {
        String xml = """
                <mujoco>
                  <worldbody>
                    <site name="sphere_site" type="sphere" size="0.5" pos="1 0 0"/>
                    <site name="box_site" type="box" size="0.5 0.5 0.5" pos="0 0 0"/>
                  </worldbody>
                </mujoco>
                """;
        try (MjModel m = MjModel.loadFromXmlString(xml);
             MjData d = m.makeData()) {
            MjSim.forward(m, d);
            int sphereId = m.name2id(MjObjType.SITE, "sphere_site");
            int boxId = m.name2id(MjObjType.SITE, "box_site");

            assertTrue(MjSim.insideSite(m, d, sphereId, new double[]{1.0, 0.0, 0.0}));
            assertTrue(MjSim.insideSite(m, d, sphereId, new double[]{1.4, 0.0, 0.0}));
            assertFalse(MjSim.insideSite(m, d, sphereId, new double[]{2.0, 0.0, 0.0}));

            assertTrue(MjSim.insideSite(m, d, boxId, new double[]{0.0, 0.0, 0.0}));
            assertTrue(MjSim.insideSite(m, d, boxId, new double[]{0.4, 0.4, 0.4}));
            assertFalse(MjSim.insideSite(m, d, boxId, new double[]{1.0, 0.0, 0.0}));
        }
    }

    @Test
    void testCtrlAndSensorDelayHistory() {
        String xml = """
                <mujoco>
                  <worldbody>
                    <body>
                      <geom type="sphere" size="0.1"/>
                      <joint name="hinge" type="hinge"/>
                      <site name="site"/>
                    </body>
                  </worldbody>
                  <actuator>
                    <position name="actuator" joint="hinge" delay="0.01" nsample="4"/>
                  </actuator>
                  <sensor>
                    <accelerometer name="accel" site="site" delay="0.01" nsample="3"/>
                  </sensor>
                </mujoco>
                """;
        try (MjModel m = MjModel.loadFromXmlString(xml);
             MjData d = m.makeData()) {
            MjSim.forward(m, d);
            assertEquals(4, m.actuatorHistory().get(0));
            double[] ctrlTimes = {0.0, 0.01, 0.02, 0.03};
            double[] ctrlValues = {1.0, 2.0, 3.0, 4.0};
            MjSim.initCtrlHistory(m, d, 0, ctrlTimes, ctrlValues);

            double[] ctrlRes = new double[1];
            MjSim.readCtrl(m, d, 0, 0.02, ctrlRes, 0);
            assertEquals(2.0, ctrlRes[0], 0.0);

            assertEquals(3, m.sensorHistory().get(0));
            assertEquals(3, m.sensorDim().get(0));
            double[] sensTimes = {0.0, 0.01, 0.02};
            double[] sensValues = {1, 2, 3, 4, 5, 6, 7, 8, 9};
            MjSim.initSensorHistory(m, d, 0, sensTimes, sensValues, 0.0);

            double[] sensRes = new double[3];
            MjSim.readSensor(m, d, 0, 0.02, sensRes, 0);
            assertArrayEquals(new double[]{4.0, 5.0, 6.0}, sensRes, 0.0);
        }
    }
}
