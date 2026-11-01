package org.mujoco4j.internal.glfw;

import org.mujoco4j.MjException;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Pure Java 25 FFM ({@code java.lang.foreign}) bindings for GLFW 3.4.
 */
public final class GlfwBindings {

    public static final int GLFW_TRUE = 1;
    public static final int GLFW_FALSE = 0;

    public static final int GLFW_PRESS = 1;
    public static final int GLFW_RELEASE = 0;

    public static final int GLFW_MOUSE_BUTTON_LEFT = 0;
    public static final int GLFW_MOUSE_BUTTON_RIGHT = 1;
    public static final int GLFW_MOUSE_BUTTON_MIDDLE = 2;

    public static final int GLFW_KEY_SPACE = 32;
    public static final int GLFW_KEY_ESCAPE = 256;
    public static final int GLFW_KEY_BACKSPACE = 259;

    public static final int GLFW_MOD_SHIFT = 0x0001;
    public static final int GLFW_MOD_CONTROL = 0x0002;
    public static final int GLFW_MOD_ALT = 0x0004;

    public static final int GLFW_SAMPLES = 0x0002100D;
    public static final int GLFW_VISIBLE = 0x00020004;

    private static final Linker LINKER = Linker.nativeLinker();
    private static final AtomicBoolean LOADED = new AtomicBoolean(false);
    private static volatile SymbolLookup LOOKUP;

    private static MethodHandle mhInit;
    private static MethodHandle mhTerminate;
    private static MethodHandle mhWindowHint;
    private static MethodHandle mhCreateWindow;
    private static MethodHandle mhDestroyWindow;
    private static MethodHandle mhWindowShouldClose;
    private static MethodHandle mhSetWindowShouldClose;
    private static MethodHandle mhMakeContextCurrent;
    private static MethodHandle mhSwapBuffers;
    private static MethodHandle mhSwapInterval;
    private static MethodHandle mhPollEvents;
    private static MethodHandle mhGetFramebufferSize;
    private static MethodHandle mhGetWindowSize;
    private static MethodHandle mhGetKey;
    private static MethodHandle mhSetCursorPosCallback;
    private static MethodHandle mhSetMouseButtonCallback;
    private static MethodHandle mhSetScrollCallback;
    private static MethodHandle mhSetKeyCallback;

    public static final FunctionDescriptor CURSOR_POS_CB_DESC = FunctionDescriptor.ofVoid(
            ValueLayout.ADDRESS, ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE
    );
    public static final FunctionDescriptor MOUSE_BUTTON_CB_DESC = FunctionDescriptor.ofVoid(
            ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT
    );
    public static final FunctionDescriptor SCROLL_CB_DESC = FunctionDescriptor.ofVoid(
            ValueLayout.ADDRESS, ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_DOUBLE
    );
    public static final FunctionDescriptor KEY_CB_DESC = FunctionDescriptor.ofVoid(
            ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT
    );

    private GlfwBindings() {
    }

    public static void ensureLoaded() {
        if (LOADED.get()) {
            return;
        }
        synchronized (GlfwBindings.class) {
            if (LOADED.get()) {
                return;
            }
            loadGlfwLibrary();
            bindFunctions();
            LOADED.set(true);
        }
    }

    private static void loadGlfwLibrary() {
        List<Path> candidates = new ArrayList<>();
        String prop = System.getProperty("mujoco4j.glfw.path");
        if (prop != null && !prop.isBlank()) {
            candidates.add(Path.of(prop).toAbsolutePath().normalize());
        }
        String env = System.getenv("GLFW_LIB_PATH");
        if (env != null && !env.isBlank()) {
            candidates.add(Path.of(env).toAbsolutePath().normalize());
        }

        String fileName = mapGlfwFileName();
        Path cwd = Path.of("").toAbsolutePath();
        candidates.add(cwd.resolve(".mujoco").resolve("bin").resolve(fileName));
        candidates.add(cwd.resolve(".mujoco").resolve("lib").resolve(fileName));
        if (cwd.getParent() != null) {
            candidates.add(cwd.getParent().resolve(".mujoco").resolve("bin").resolve(fileName));
            candidates.add(cwd.getParent().resolve(".mujoco").resolve("lib").resolve(fileName));
        }

        for (Path candidate : candidates) {
            if (Files.isRegularFile(candidate)) {
                System.load(candidate.toString());
                LOOKUP = SymbolLookup.loaderLookup();
                return;
            }
        }

        try {
            System.loadLibrary("glfw3");
            LOOKUP = SymbolLookup.loaderLookup();
        } catch (UnsatisfiedLinkError e) {
            try {
                System.loadLibrary("glfw");
                LOOKUP = SymbolLookup.loaderLookup();
            } catch (UnsatisfiedLinkError e2) {
                throw new MjException("Unable to locate GLFW native library (glfw3.dll / libglfw). " +
                        "Run 'jbang scripts/SetupToolchain.java' or set -Dmujoco4j.glfw.path.", e2);
            }
        }
    }

    private static MethodHandle downcall(String name, FunctionDescriptor desc) {
        MemorySegment addr = LOOKUP.find(name)
                .orElseThrow(() -> new MjException("Missing GLFW symbol: " + name));
        return LINKER.downcallHandle(addr, desc);
    }

    private static void bindFunctions() {
        mhInit = downcall("glfwInit", FunctionDescriptor.of(ValueLayout.JAVA_INT));
        mhTerminate = downcall("glfwTerminate", FunctionDescriptor.ofVoid());
        mhWindowHint = downcall("glfwWindowHint", FunctionDescriptor.ofVoid(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));
        mhCreateWindow = downcall("glfwCreateWindow", FunctionDescriptor.of(
                ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        mhDestroyWindow = downcall("glfwDestroyWindow", FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));
        mhWindowShouldClose = downcall("glfwWindowShouldClose", FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        mhSetWindowShouldClose = downcall("glfwSetWindowShouldClose", FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.JAVA_INT));
        mhMakeContextCurrent = downcall("glfwMakeContextCurrent", FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));
        mhSwapBuffers = downcall("glfwSwapBuffers", FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));
        mhSwapInterval = downcall("glfwSwapInterval", FunctionDescriptor.ofVoid(ValueLayout.JAVA_INT));
        mhPollEvents = downcall("glfwPollEvents", FunctionDescriptor.ofVoid());
        mhGetFramebufferSize = downcall("glfwGetFramebufferSize", FunctionDescriptor.ofVoid(
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        mhGetWindowSize = downcall("glfwGetWindowSize", FunctionDescriptor.ofVoid(
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        mhGetKey = downcall("glfwGetKey", FunctionDescriptor.of(
                ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT));
        mhSetCursorPosCallback = downcall("glfwSetCursorPosCallback", FunctionDescriptor.of(
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        mhSetMouseButtonCallback = downcall("glfwSetMouseButtonCallback", FunctionDescriptor.of(
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        mhSetScrollCallback = downcall("glfwSetScrollCallback", FunctionDescriptor.of(
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
        mhSetKeyCallback = downcall("glfwSetKeyCallback", FunctionDescriptor.of(
                ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
    }

    public static int glfwInit() {
        ensureLoaded();
        try {
            return (int) mhInit.invokeExact();
        } catch (Throwable t) {
            throw new MjException("glfwInit failed", t);
        }
    }

    public static void glfwTerminate() {
        ensureLoaded();
        try {
            mhTerminate.invokeExact();
        } catch (Throwable t) {
            throw new MjException("glfwTerminate failed", t);
        }
    }

    public static void glfwWindowHint(int hint, int value) {
        ensureLoaded();
        try {
            mhWindowHint.invokeExact(hint, value);
        } catch (Throwable t) {
            throw new MjException("glfwWindowHint failed", t);
        }
    }

    public static MemorySegment glfwCreateWindow(int width, int height, String title) {
        ensureLoaded();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment titleSeg = arena.allocateFrom(title != null ? title : "MuJoCo");
            return (MemorySegment) mhCreateWindow.invokeExact(
                    width, height, titleSeg, MemorySegment.NULL, MemorySegment.NULL);
        } catch (Throwable t) {
            throw new MjException("glfwCreateWindow failed", t);
        }
    }

    public static void glfwDestroyWindow(MemorySegment window) {
        ensureLoaded();
        try {
            mhDestroyWindow.invokeExact(window);
        } catch (Throwable t) {
            throw new MjException("glfwDestroyWindow failed", t);
        }
    }

    public static boolean glfwWindowShouldClose(MemorySegment window) {
        ensureLoaded();
        try {
            return ((int) mhWindowShouldClose.invokeExact(window)) != 0;
        } catch (Throwable t) {
            throw new MjException("glfwWindowShouldClose failed", t);
        }
    }

    public static void glfwSetWindowShouldClose(MemorySegment window, boolean shouldClose) {
        ensureLoaded();
        try {
            mhSetWindowShouldClose.invokeExact(window, shouldClose ? 1 : 0);
        } catch (Throwable t) {
            throw new MjException("glfwSetWindowShouldClose failed", t);
        }
    }

    public static void glfwMakeContextCurrent(MemorySegment window) {
        ensureLoaded();
        try {
            mhMakeContextCurrent.invokeExact(window);
        } catch (Throwable t) {
            throw new MjException("glfwMakeContextCurrent failed", t);
        }
    }

    public static void glfwSwapBuffers(MemorySegment window) {
        ensureLoaded();
        try {
            mhSwapBuffers.invokeExact(window);
        } catch (Throwable t) {
            throw new MjException("glfwSwapBuffers failed", t);
        }
    }

    public static void glfwSwapInterval(int interval) {
        ensureLoaded();
        try {
            mhSwapInterval.invokeExact(interval);
        } catch (Throwable t) {
            throw new MjException("glfwSwapInterval failed", t);
        }
    }

    public static void glfwPollEvents() {
        ensureLoaded();
        try {
            mhPollEvents.invokeExact();
        } catch (Throwable t) {
            throw new MjException("glfwPollEvents failed", t);
        }
    }

    public static int[] glfwGetFramebufferSize(MemorySegment window) {
        ensureLoaded();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment w = arena.allocate(ValueLayout.JAVA_INT);
            MemorySegment h = arena.allocate(ValueLayout.JAVA_INT);
            mhGetFramebufferSize.invokeExact(window, w, h);
            return new int[]{w.get(ValueLayout.JAVA_INT, 0), h.get(ValueLayout.JAVA_INT, 0)};
        } catch (Throwable t) {
            throw new MjException("glfwGetFramebufferSize failed", t);
        }
    }

    public static int[] glfwGetWindowSize(MemorySegment window) {
        ensureLoaded();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment w = arena.allocate(ValueLayout.JAVA_INT);
            MemorySegment h = arena.allocate(ValueLayout.JAVA_INT);
            mhGetWindowSize.invokeExact(window, w, h);
            return new int[]{w.get(ValueLayout.JAVA_INT, 0), h.get(ValueLayout.JAVA_INT, 0)};
        } catch (Throwable t) {
            throw new MjException("glfwGetWindowSize failed", t);
        }
    }

    public static int glfwGetKey(MemorySegment window, int key) {
        ensureLoaded();
        try {
            return (int) mhGetKey.invokeExact(window, key);
        } catch (Throwable t) {
            throw new MjException("glfwGetKey failed", t);
        }
    }

    public static void glfwSetCursorPosCallback(MemorySegment window, MemorySegment cbStub) {
        ensureLoaded();
        try {
            MemorySegment ignored = (MemorySegment) mhSetCursorPosCallback.invokeExact(window, cbStub);
        } catch (Throwable t) {
            throw new MjException("glfwSetCursorPosCallback failed", t);
        }
    }

    public static void glfwSetMouseButtonCallback(MemorySegment window, MemorySegment cbStub) {
        ensureLoaded();
        try {
            MemorySegment ignored = (MemorySegment) mhSetMouseButtonCallback.invokeExact(window, cbStub);
        } catch (Throwable t) {
            throw new MjException("glfwSetMouseButtonCallback failed", t);
        }
    }

    public static void glfwSetScrollCallback(MemorySegment window, MemorySegment cbStub) {
        ensureLoaded();
        try {
            MemorySegment ignored = (MemorySegment) mhSetScrollCallback.invokeExact(window, cbStub);
        } catch (Throwable t) {
            throw new MjException("glfwSetScrollCallback failed", t);
        }
    }

    public static void glfwSetKeyCallback(MemorySegment window, MemorySegment cbStub) {
        ensureLoaded();
        try {
            MemorySegment ignored = (MemorySegment) mhSetKeyCallback.invokeExact(window, cbStub);
        } catch (Throwable t) {
            throw new MjException("glfwSetKeyCallback failed", t);
        }
    }

    private static String mapGlfwFileName() {
        String os = System.getProperty("os.name", "").toLowerCase();
        if (os.contains("win")) {
            return "glfw3.dll";
        } else if (os.contains("mac") || os.contains("darwin")) {
            return "libglfw.dylib";
        } else {
            return "libglfw.so.3";
        }
    }
}
