package org.mujoco4j;

import org.mujoco4j.internal.panama.mujoco_h;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Handles deterministic native library discovery, loading, plugin registration,
 * and ABI version verification for MuJoCo 3.17.0.
 */
public final class MjNativeLoader {

    private static final AtomicBoolean LOADED = new AtomicBoolean(false);
    private static volatile Path loadedLibraryPath;

    private MjNativeLoader() {
    }

    /**
     * Ensures that the MuJoCo native library is loaded into the current ClassLoader,
     * verifies that the runtime library version matches {@code mjVERSION_HEADER},
     * loads bundled MuJoCo plugins if present, and installs the default timer callback.
     */
    public static void ensureLoaded() {
        if (LOADED.get()) {
            return;
        }
        synchronized (MjNativeLoader.class) {
            if (LOADED.get()) {
                return;
            }
            loadNativeLibrary();
            verifyVersion();
            loadPluginsIfPresent();
            MjCallbacks.installDefaultTimerIfNeeded();
            LOADED.set(true);
        }
    }

    public static Path loadedLibraryPath() {
        ensureLoaded();
        return loadedLibraryPath;
    }

    public static int version() {
        ensureLoaded();
        return mujoco_h.mj_version();
    }

    public static String versionString() {
        ensureLoaded();
        MemorySegment seg = mujoco_h.mj_versionString();
        return seg.reinterpret(256).getString(0);
    }

    public static int headerVersion() {
        return mujoco_h.mjVERSION_HEADER();
    }

    private static void loadNativeLibrary() {
        List<Path> candidates = new ArrayList<>();

        String propPath = System.getProperty("mujoco4j.library.path");
        if (propPath != null && !propPath.isBlank()) {
            candidates.add(Path.of(propPath).toAbsolutePath().normalize());
        }

        String envPath = System.getenv("MUJOCO_LIB_PATH");
        if (envPath != null && !envPath.isBlank()) {
            candidates.add(Path.of(envPath).toAbsolutePath().normalize());
        }

        String libFileName = mapLibraryFileName();
        Path cwd = Path.of("").toAbsolutePath();
        candidates.add(cwd.resolve(".mujoco").resolve("bin").resolve(libFileName));
        candidates.add(cwd.resolve(".mujoco").resolve("lib").resolve(libFileName));
        if (cwd.getParent() != null) {
            candidates.add(cwd.getParent().resolve(".mujoco").resolve("bin").resolve(libFileName));
            candidates.add(cwd.getParent().resolve(".mujoco").resolve("lib").resolve(libFileName));
        }

        for (Path candidate : candidates) {
            if (Files.isRegularFile(candidate)) {
                System.load(candidate.toString());
                loadedLibraryPath = candidate;
                return;
            }
        }

        // Fallback to standard java.library.path
        System.loadLibrary("mujoco");
    }

    private static void verifyVersion() {
        int runtimeVer = mujoco_h.mj_version();
        int expectedVer = mujoco_h.mjVERSION_HEADER();
        if (runtimeVer != expectedVer) {
            throw new MjException(
                    "MuJoCo native library ABI version mismatch: header expects "
                            + expectedVer + " but loaded DLL reported " + runtimeVer);
        }
    }

    private static void loadPluginsIfPresent() {
        if (loadedLibraryPath == null) {
            return;
        }
        Path pluginDir = loadedLibraryPath.getParent().resolve("mujoco_plugin");
        if (Files.isDirectory(pluginDir)) {
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment dirStr = arena.allocateFrom(pluginDir.toAbsolutePath().toString());
                mujoco_h.mj_loadAllPluginLibraries(dirStr, MemorySegment.NULL);
            }
        }
    }

    private static String mapLibraryFileName() {
        String os = System.getProperty("os.name", "").toLowerCase();
        if (os.contains("win")) {
            return "mujoco.dll";
        } else if (os.contains("mac") || os.contains("darwin")) {
            return "libmujoco.dylib";
        } else {
            return "libmujoco.so";
        }
    }
}
