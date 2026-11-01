# mujoco4j

Pure **Java 25 LTS** bindings and idiomatic physics simulation library for **DeepMind MuJoCo 3.17.0** (`mjVERSION_HEADER = 3017000`), built on the **Foreign Function & Memory (FFM) API (`java.lang.foreign` / Project Panama)** with zero JNI or C++ glue code.

---

## Features

- **Zero JNI / Zero Native Glue:** Directly invokes MuJoCo C functions via `java.lang.foreign.Linker` and accesses off-heap simulation state via `MemorySegment` views.
- **Two-Layer Architecture:**
  - **Layer 1 (`org.mujoco4j.internal.panama`):** Low-level FFM bindings generated via `jextract` from MuJoCo 3.17.0 headers (`mujoco.h`).
  - **Layer 2 (`org.mujoco4j`):** High-level, memory-safe, `AutoCloseable` Java 25 API (`MjModel`, `MjData`, `MjSim`, `MjMath`, `MjRay`, `MjCallbacks`, `MjVfs`, zero-copy primitive vector views, and contact/element sub-struct views).
- **Memory & Thread Safety:**
  - Deterministic off-heap lifecycle with `try-with-resources` (`MjModel`, `MjData`, `MjVfs`), preventing double-free or use-after-close bugs.
  - Bounds-checked zero-copy vector views (`MjDoubleVector`, `MjIntVector`, `MjFloatVector`, `MjLongVector`, `MjByteVector`).
  - Safe upcall trampolines (`MjCallbacks`) that catch Java exceptions before the C boundary and rethrow them cleanly on the Java thread.
- **Verified Against DeepMind's Official Test Suite:**
  - **129 / 129 black-box tests passing**, including ports of DeepMind's `bindings_test.py`, all 91 official XML models in `.mujoco/model/`, and physics invariants (free-fall kinematics, symplectic Euler & RK4 energy conservation, contact equilibrium, PD/motor actuator control, kinematics/sensors, and state determinism).

---

## Quick Start

### 1. Setup Local Native Toolchain & MuJoCo 3.17.0 Distribution

Run the self-bootstrapping setup script using [JBang](https://www.jbang.dev/) (or `java`):

```powershell
jbang scripts/SetupToolchain.java
```

This downloads and verifies:
- MuJoCo `3.17.0` (`bin/mujoco.dll`, `include/mujoco/*.h`, `model/**`) into `.mujoco/`
- DeepMind's official test suites into `.mujoco/deepmind-tests/`

### 2. Build & Run All 129 Black-Box Tests

```powershell
.\mvnw.cmd test
```

### 3. Install to Local Maven Repository & Run Standalone JBang Script

```powershell
.\mvnw.cmd install -DskipTests
jbang --fresh run scripts/HelloMuJoCo.java
```

---

## Example Usage

```java
import org.mujoco4j.*;

public class Example {
    public static void main(String[] args) {
        String xml = """
            <mujoco model="falling_ball">
              <option timestep="0.002" gravity="0 0 -9.81"/>
              <worldbody>
                <geom name="floor" type="plane" size="2 2 0.1"/>
                <body name="ball" pos="0 0 1.0">
                  <freejoint/>
                  <geom name="ball_geom" type="sphere" size="0.1" mass="1.0"/>
                </body>
              </worldbody>
            </mujoco>
            """;

        try (MjModel model = MjModel.loadFromXmlString(xml);
             MjData data = model.makeData()) {

            for (int step = 0; step < 500; step++) {
                MjSim.step(model, data);
            }

            System.out.printf("Time: %.3f s | Ball z: %.4f m | Contacts: %d%n",
                data.time(), data.qpos().get(2), data.ncon());
        }
    }
}
```
