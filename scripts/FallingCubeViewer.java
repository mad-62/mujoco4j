///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25+
//RUNTIME_OPTIONS --enable-native-access=ALL-UNNAMED
//DEPS org.mujoco4j:mujoco4j:0.1.0-SNAPSHOT

import org.mujoco4j.MjData;
import org.mujoco4j.MjModel;
import org.mujoco4j.MjSim;
import org.mujoco4j.MjViewer;

/**
 * Direct Java 25 equivalent of the Python mujoco.viewer.launch_passive demo.
 *
 * Run via:
 *   jbang --fresh run scripts/FallingCubeViewer.java
 */
public class FallingCubeViewer {

    public static void main(String[] args) throws InterruptedException {
        String xml = """
                <mujoco>
                    <option timestep="0.002"/>

                    <worldbody>
                        <light pos="0 0 3"/>

                        <geom name="floor"
                              type="plane"
                              size="5 5 0.1"/>

                        <body name="cube" pos="0 0 2.0">
                            <freejoint/>
                            <geom type="box"
                                  size="0.1 0.1 0.1"
                                  mass="1"/>
                        </body>
                    </worldbody>
                </mujoco>
                """;

        try (MjModel model = MjModel.loadFromXmlString(xml);
             MjData data = model.makeData();
             MjViewer viewer = MjViewer.launchPassive(model, data)) {

            while (viewer.isRunning()) {
                long stepStart = System.nanoTime();

                MjSim.step(model, data);
                viewer.sync();

                double elapsedSec = (System.nanoTime() - stepStart) / 1_000_000_000.0;
                double timeUntilNextStep = model.opt().timestep() - elapsedSec;
                if (timeUntilNextStep > 0) {
                    Thread.sleep((long) (timeUntilNextStep * 2.5 * 1000.0));
                }
            }
        }
    }
}
