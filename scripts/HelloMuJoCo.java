///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//RUNTIME_OPTIONS --enable-native-access=ALL-UNNAMED
//DEPS org.mujoco4j:mujoco4j:0.1.0-SNAPSHOT

import org.mujoco4j.*;

import java.nio.file.Path;

/**
 * Self-contained JBang verification script demonstrating end-to-end simulation of
 * Google DeepMind's official MuJoCo 3.17.0 humanoid model via {@code org.mujoco4j}.
 */
public class HelloMuJoCo {

    public static void main(String[] args) {
        System.out.println("====================================================");
        System.out.println("  mujoco4j — Pure Java 25 FFM Bindings for MuJoCo  ");
        System.out.println("====================================================");
        System.out.println("MuJoCo Native Version : " + MjSim.versionString() + " (" + MjSim.version() + ")");

        Path humanoidXml = Path.of(".mujoco", "model", "humanoid", "humanoid.xml");
        try (MjModel model = MjModel.loadFromXmlPath(humanoidXml);
             MjData data = model.makeData()) {

            System.out.println("Loaded Model          : " + model.modelName());
            System.out.println("Bodies / Joints / DoFs: " + model.nbody() + " / " + model.njnt() + " / " + model.nv());
            System.out.println("Actuators / Sensors   : " + model.nu() + " / " + model.nsensor());
            System.out.printf("Total Mass            : %.4f kg%n", model.totalMass());

            MjSim.forward(model, data);
            double initialTorsoZ = data.body("torso").xpos().get(2);
            System.out.printf("Initial Torso Z       : %.6f m%n", initialTorsoZ);

            int steps = 500;
            for (int step = 0; step < steps; step++) {
                MjSim.step(model, data);
            }

            double finalTorsoZ = data.body("torso").xpos().get(2);
            System.out.printf("Simulated %d steps   : t = %.3f s, Torso Z = %.6f m, Active Contacts = %d%n",
                    steps, data.time(), finalTorsoZ, data.ncon());
        }
    }
}
