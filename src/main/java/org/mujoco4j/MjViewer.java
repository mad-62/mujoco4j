package org.mujoco4j;

import org.mujoco4j.internal.glfw.GlfwBindings;
import org.mujoco4j.internal.panama.mujoco_h;

import java.lang.foreign.Arena;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Interactive 3D OpenGL viewer for MuJoCo simulations, powered by pure Java 25 FFM
 * ({@code java.lang.foreign}), GLFW 3.4, and MuJoCo's native {@code mjr_render} engine.
 *
 * <p>Provides the Java equivalent of Python's {@code mujoco.viewer.launch_passive(model, data)}:
 * <pre>{@code
 * try (MjViewer viewer = MjViewer.launchPassive(model, data)) {
 *     while (viewer.isRunning()) {
 *         MjSim.step(model, data);
 *         viewer.sync();
 *     }
 * }
 * }</pre>
 */
public final class MjViewer implements AutoCloseable {

    private final MjModel model;
    private final MjData data;
    private final Arena callbackArena;
    private final MemorySegment window;
    private final MjvScene scene;
    private final MjvCamera camera;
    private final MjvOption option;
    private final MjvPerturb perturb;
    private final MjrContext context;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    private boolean buttonLeft = false;
    private boolean buttonMiddle = false;
    private boolean buttonRight = false;
    private double lastX = 0.0;
    private double lastY = 0.0;

    private MjViewer(MjModel model, MjData data, String title, int width, int height) {
        this.model = Objects.requireNonNull(model, "model must not be null");
        this.data = Objects.requireNonNull(data, "data must not be null");
        model.checkOpen();
        data.checkOpen();

        MjNativeLoader.ensureLoaded();
        GlfwBindings.ensureLoaded();

        if (GlfwBindings.glfwInit() == GlfwBindings.GLFW_FALSE) {
            throw new MjException("Failed to initialize GLFW");
        }

        GlfwBindings.glfwWindowHint(GlfwBindings.GLFW_SAMPLES, 4);
        this.window = GlfwBindings.glfwCreateWindow(width, height, title);
        if (this.window == null || this.window.equals(MemorySegment.NULL)) {
            GlfwBindings.glfwTerminate();
            throw new MjException("Failed to create GLFW OpenGL window");
        }

        GlfwBindings.glfwMakeContextCurrent(this.window);
        GlfwBindings.glfwSwapInterval(1);

        this.camera = new MjvCamera();
        mujoco_h.mjv_defaultFreeCamera(model.segment(), this.camera.segment());
        this.option = new MjvOption();
        this.perturb = new MjvPerturb();
        this.scene = new MjvScene(model, Math.max(2000, model.ngeom() + 500));
        this.context = new MjrContext(model, MjrContext.FONTSCALE_150);

        this.callbackArena = Arena.ofShared();
        installInputCallbacks();

        // Render initial frame immediately
        MjSim.forward(model, data);
        sync();
    }

    /**
     * Launches a passive interactive 3D MuJoCo viewer window (1280x720).
     */
    public static MjViewer launchPassive(MjModel model, MjData data) {
        return launchPassive(model, data, "MuJoCo4J Viewer (MuJoCo " + MjNativeLoader.versionString() + ")", 1280, 720);
    }

    /**
     * Launches a passive interactive 3D MuJoCo viewer window with custom title and dimensions.
     */
    public static MjViewer launchPassive(MjModel model, MjData data, String title, int width, int height) {
        return new MjViewer(model, data, title, width, height);
    }

    /**
     * Returns {@code true} while the viewer window is open and has not been requested to close.
     */
    public boolean isRunning() {
        if (closed.get()) {
            return false;
        }
        return !GlfwBindings.glfwWindowShouldClose(window);
    }

    /**
     * Synchronizes the 3D scene with the current state in {@code model} and {@code data},
     * renders the frame via MuJoCo's OpenGL renderer ({@code mjr_render}), swaps buffers,
     * and polls window/mouse/keyboard events.
     */
    public void sync() {
        checkOpen();
        model.checkOpen();
        data.checkOpen();

        GlfwBindings.glfwMakeContextCurrent(window);
        int[] fbSize = GlfwBindings.glfwGetFramebufferSize(window);
        int width = fbSize[0];
        int height = fbSize[1];

        if (width > 0 && height > 0) {
            scene.updateScene(model, data, option, perturb, camera, MjCatBit.ALL.value());
            context.render(0, 0, width, height, scene);
        }

        GlfwBindings.glfwSwapBuffers(window);
        GlfwBindings.glfwPollEvents();
    }

    public MjvCamera camera() {
        checkOpen();
        return camera;
    }

    public MjvOption option() {
        checkOpen();
        return option;
    }

    public MjvScene scene() {
        checkOpen();
        return scene;
    }

    private void checkOpen() {
        if (closed.get()) {
            throw new IllegalStateException("MjViewer has already been closed.");
        }
    }

    private void installInputCallbacks() {
        try {
            Linker linker = Linker.nativeLinker();
            MethodHandles.Lookup lookup = MethodHandles.lookup();

            MethodHandle cursorMh = lookup.findVirtual(MjViewer.class, "onCursorPos",
                    MethodType.methodType(void.class, MemorySegment.class, double.class, double.class))
                    .bindTo(this);
            MemorySegment cursorStub = linker.upcallStub(cursorMh, GlfwBindings.CURSOR_POS_CB_DESC, callbackArena);
            GlfwBindings.glfwSetCursorPosCallback(window, cursorStub);

            MethodHandle buttonMh = lookup.findVirtual(MjViewer.class, "onMouseButton",
                    MethodType.methodType(void.class, MemorySegment.class, int.class, int.class, int.class))
                    .bindTo(this);
            MemorySegment buttonStub = linker.upcallStub(buttonMh, GlfwBindings.MOUSE_BUTTON_CB_DESC, callbackArena);
            GlfwBindings.glfwSetMouseButtonCallback(window, buttonStub);

            MethodHandle scrollMh = lookup.findVirtual(MjViewer.class, "onScroll",
                    MethodType.methodType(void.class, MemorySegment.class, double.class, double.class))
                    .bindTo(this);
            MemorySegment scrollStub = linker.upcallStub(scrollMh, GlfwBindings.SCROLL_CB_DESC, callbackArena);
            GlfwBindings.glfwSetScrollCallback(window, scrollStub);

            MethodHandle keyMh = lookup.findVirtual(MjViewer.class, "onKey",
                    MethodType.methodType(void.class, MemorySegment.class, int.class, int.class, int.class, int.class))
                    .bindTo(this);
            MemorySegment keyStub = linker.upcallStub(keyMh, GlfwBindings.KEY_CB_DESC, callbackArena);
            GlfwBindings.glfwSetKeyCallback(window, keyStub);
        } catch (ReflectiveOperationException e) {
            throw new MjException("Failed to bind GLFW input callbacks", e);
        }
    }

    private void onCursorPos(MemorySegment win, double xpos, double ypos) {
        if (!buttonLeft && !buttonMiddle && !buttonRight) {
            lastX = xpos;
            lastY = ypos;
            return;
        }
        double dx = xpos - lastX;
        double dy = ypos - lastY;
        lastX = xpos;
        lastY = ypos;

        int[] winSize = GlfwBindings.glfwGetWindowSize(window);
        int height = Math.max(1, winSize[1]);

        boolean shift = GlfwBindings.glfwGetKey(window, 340) == GlfwBindings.GLFW_PRESS
                || GlfwBindings.glfwGetKey(window, 344) == GlfwBindings.GLFW_PRESS;

        int action;
        if (buttonRight) {
            action = shift ? mujoco_h.mjMOUSE_MOVE_H() : mujoco_h.mjMOUSE_MOVE_V();
        } else if (buttonLeft) {
            action = shift ? mujoco_h.mjMOUSE_ROTATE_H() : mujoco_h.mjMOUSE_ROTATE_V();
        } else {
            action = mujoco_h.mjMOUSE_ZOOM();
        }

        if (model.isOpen()) {
            mujoco_h.mjv_moveCamera(model.segment(), action, dx / height, dy / height, camera.segment());
        }
    }

    private void onMouseButton(MemorySegment win, int button, int act, int mods) {
        boolean pressed = (act == GlfwBindings.GLFW_PRESS);
        if (button == GlfwBindings.GLFW_MOUSE_BUTTON_LEFT) {
            buttonLeft = pressed;
        } else if (button == GlfwBindings.GLFW_MOUSE_BUTTON_MIDDLE) {
            buttonMiddle = pressed;
        } else if (button == GlfwBindings.GLFW_MOUSE_BUTTON_RIGHT) {
            buttonRight = pressed;
        }
    }

    private void onScroll(MemorySegment win, double xoffset, double yoffset) {
        if (model.isOpen()) {
            mujoco_h.mjv_moveCamera(model.segment(), mujoco_h.mjMOUSE_ZOOM(), 0.0, -0.05 * yoffset, camera.segment());
        }
    }

    private void onKey(MemorySegment win, int key, int scancode, int act, int mods) {
        if (act == GlfwBindings.GLFW_PRESS) {
            if (key == GlfwBindings.GLFW_KEY_ESCAPE) {
                GlfwBindings.glfwSetWindowShouldClose(window, true);
            } else if (key == GlfwBindings.GLFW_KEY_BACKSPACE && model.isOpen() && data.isOpen()) {
                MjSim.resetData(model, data);
                MjSim.forward(model, data);
            }
        }
    }

    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            try {
                GlfwBindings.glfwMakeContextCurrent(window);
                context.close();
                scene.close();
                GlfwBindings.glfwDestroyWindow(window);
                callbackArena.close();
            } finally {
                GlfwBindings.glfwTerminate();
            }
        }
    }
}
