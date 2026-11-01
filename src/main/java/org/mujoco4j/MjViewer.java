package org.mujoco4j;

import org.mujoco4j.internal.glfw.GlfwBindings;
import org.mujoco4j.internal.panama.mjData_;
import org.mujoco4j.internal.panama.mjModel_;
import org.mujoco4j.internal.panama.mjOption_;
import org.mujoco4j.internal.panama.mjUI_;
import org.mujoco4j.internal.panama.mjfItemEnable;
import org.mujoco4j.internal.panama.mjrRect_;
import org.mujoco4j.internal.panama.mjuiDef_;
import org.mujoco4j.internal.panama.mjuiItem_;
import org.mujoco4j.internal.panama.mjuiState_;
import org.mujoco4j.internal.panama.mjuiThemeColor_;
import org.mujoco4j.internal.panama.mjuiThemeSpacing_;
import org.mujoco4j.internal.panama.mjvCamera_;
import org.mujoco4j.internal.panama.mjvFigure_;
import org.mujoco4j.internal.panama.mjvOption_;
import org.mujoco4j.internal.panama.mjvPerturb_;
import org.mujoco4j.internal.panama.mjvScene_;
import org.mujoco4j.internal.panama.mujoco_h;

import java.lang.foreign.Arena;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Full-featured interactive 3D OpenGL viewer for MuJoCo simulations, matching DeepMind's
 * {@code simulate} / Python {@code mujoco.viewer.launch_passive} experience:
 * <ul>
 *   <li>Collapsible Left &amp; Right {@code mjUI} sidebars ({@code Tab} / {@code Shift+Tab})</li>
 *   <li>Live HUD Info ({@code F2}) &amp; Keyboard Shortcuts Help ({@code F1}) via {@code mjr_overlay}</li>
 *   <li>Live scrolling Energy &amp; Sensor plots ({@code F4}) via {@code mjr_figure}</li>
 *   <li>Double-click 3D body selection ({@code mjv_select}) &amp; {@code Ctrl + Mouse Drag} 3D force/torque perturbation</li>
 *   <li>Interactive camera orbit, pan, zoom, tracking, pause ({@code Space}), single-step ({@code Right Arrow}), and reset ({@code Backspace})</li>
 * </ul>
 */
public final class MjViewer implements AutoCloseable {

    private static final int KEY_TAB = 258;
    private static final int KEY_RIGHT = 262;
    private static final int KEY_F1 = 290;
    private static final int KEY_F2 = 291;
    private static final int KEY_F3 = 292;
    private static final int KEY_F4 = 293;

    private final MjModel model;
    private final MjData data;
    private final Arena uiArena;
    private final MemorySegment window;
    private final MjvScene scene;
    private final MjvCamera camera;
    private final MjvOption option;
    private final MjvPerturb perturb;
    private final MjrContext context;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    // Native mjUI state & panels
    private final MemorySegment uiState;
    private final MemorySegment ui0; // Left sidebar
    private final MemorySegment ui1; // Right sidebar
    private final MemorySegment energyFig;

    // UI bound native memory cells
    private final MemorySegment runCheckPtr;      // int: 1 = running, 0 = paused
    private final MemorySegment showInfoPtr;      // int: 1 = show F2 info overlay
    private final MemorySegment showHelpPtr;      // int: 1 = show F1 help overlay
    private final MemorySegment showPlotPtr;      // int: 1 = show F4 energy plot

    private boolean showLeftUi = true;
    private boolean showRightUi = true;
    private boolean stepOnceRequested = false;
    private boolean wasPaused = false;
    private final double[] pausedStateSnapshot;

    // FPS & timing
    private long lastFpsTimestampNanos = System.nanoTime();
    private long lastRenderNanos = 0L;
    private int framesSinceLastFps = 0;
    private double measuredFps = 60.0;
    private double lastClickTimeSec = -10.0;
    private int lastClickButton = 0;
    private int lastFbW = -1;
    private int lastFbH = -1;
    private boolean lastShowLeft = false;
    private boolean lastShowRight = false;

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
        GlfwBindings.glfwSwapInterval(0);

        this.camera = new MjvCamera();
        mujoco_h.mjv_defaultFreeCamera(model.segment(), this.camera.segment());
        this.option = new MjvOption();
        this.perturb = new MjvPerturb();
        this.scene = new MjvScene(model, Math.max(3000, model.ngeom() + 1000));
        this.context = new MjrContext(model, MjrContext.FONTSCALE_100);

        int stateFloats = MjSim.stateSize(model, MjState.INTEGRATION.value());
        this.pausedStateSnapshot = new double[Math.max(1, stateFloats)];

        this.uiArena = Arena.ofShared();
        this.uiState = uiArena.allocate(mjuiState_.layout());
        this.ui0 = uiArena.allocate(mjUI_.layout());
        this.ui1 = uiArena.allocate(mjUI_.layout());
        this.energyFig = uiArena.allocate(mjvFigure_.layout());

        this.runCheckPtr = uiArena.allocateFrom(ValueLayout.JAVA_INT, 1);
        this.showInfoPtr = uiArena.allocateFrom(ValueLayout.JAVA_INT, 1);
        this.showHelpPtr = uiArena.allocateFrom(ValueLayout.JAVA_INT, 0);
        this.showPlotPtr = uiArena.allocateFrom(ValueLayout.JAVA_INT, 1);

        initFigure();
        initUiPanels();
        installInputCallbacks();

        MjSim.forward(model, data);
        updateRectanglesAndResizeUi();
        sync();
    }

    /**
     * Launches a passive interactive 3D MuJoCo viewer window (1440x850).
     */
    public static MjViewer launchPassive(MjModel model, MjData data) {
        return launchPassive(model, data, "MuJoCo4J Studio Viewer (MuJoCo " + MjNativeLoader.versionString() + ")", 1440, 850);
    }

    /**
     * Launches a passive interactive 3D MuJoCo viewer window with custom title and dimensions.
     */
    public static MjViewer launchPassive(MjModel model, MjData data, String title, int width, int height) {
        return new MjViewer(model, data, title, width, height);
    }

    public boolean isRunning() {
        if (closed.get()) {
            return false;
        }
        return !GlfwBindings.glfwWindowShouldClose(window);
    }

    public boolean isPaused() {
        return runCheckPtr.get(ValueLayout.JAVA_INT, 0) == 0;
    }

    public void setPaused(boolean paused) {
        runCheckPtr.set(ValueLayout.JAVA_INT, 0, paused ? 0 : 1);
    }

    /**
     * Synchronizes the 3D scene, applies any interactive 3D force perturbations or pause state,
     * renders the 3D viewport, HUD overlays, live energy graph, and collapsible {@code mjUI} sidebars.
     */
    public void sync() {
        checkOpen();
        model.checkOpen();
        data.checkOpen();

        GlfwBindings.glfwMakeContextCurrent(window);

        // Handle Pause / Single-Step / Interactive Perturbations seamlessly
        boolean paused = isPaused();
        if (paused) {
            if (!wasPaused) {
                MjSim.getState(model, data, pausedStateSnapshot, MjState.INTEGRATION.value());
                wasPaused = true;
            } else if (stepOnceRequested) {
                stepOnceRequested = false;
                MjSim.getState(model, data, pausedStateSnapshot, MjState.INTEGRATION.value());
            } else {
                MjSim.setState(model, data, pausedStateSnapshot, MjState.INTEGRATION.value());
                if (mjvPerturb_.active(perturb.segment()) != 0) {
                    mujoco_h.mjv_applyPerturbPose(model.segment(), data.segment(), perturb.segment(), 1);
                }
                MjSim.forward(model, data);
                MjSim.getState(model, data, pausedStateSnapshot, MjState.INTEGRATION.value());
            }
        } else {
            wasPaused = false;
            // Clear previous perturbation forces and apply current active 3D mouse pull/torque
            data.xfrcApplied().fill(0.0);
            if (mjvPerturb_.active(perturb.segment()) != 0) {
                mujoco_h.mjv_applyPerturbForce(model.segment(), data.segment(), perturb.segment());
            }
        }

        // Throttle actual OpenGL rendering to ~60 FPS (16.6ms) just like Python's launch_passive,
        // so high-frequency physics loops (e.g., 500Hz) are never slowed down by VSync or UI redraws.
        long now = System.nanoTime();
        if (!paused && (now - lastRenderNanos) < 16_000_000L) {
            GlfwBindings.glfwPollEvents();
            return;
        }
        lastRenderNanos = now;

        // Update FPS counter
        framesSinceLastFps++;
        double elapsedSec = (now - lastFpsTimestampNanos) / 1_000_000_000.0;
        if (elapsedSec >= 0.25) {
            measuredFps = framesSinceLastFps / elapsedSec;
            framesSinceLastFps = 0;
            lastFpsTimestampNanos = now;
        }

        updateRectanglesAndResizeUi();

        MemorySegment rects = mjuiState_.rect(uiState);
        MemorySegment viewport3d = mjrRect_.asSlice(rects, 3);
        int vpLeft = mjrRect_.left(viewport3d);
        int vpBottom = mjrRect_.bottom(viewport3d);
        int vpWidth = mjrRect_.width(viewport3d);
        int vpHeight = mjrRect_.height(viewport3d);

        if (vpWidth > 0 && vpHeight > 0) {
            scene.updateScene(model, data, option, perturb, camera, MjCatBit.ALL.value());
            context.render(vpLeft, vpBottom, vpWidth, vpHeight, scene);

            // Render HUD overlays (Info F2 & Help F1)
            renderHudOverlays(viewport3d);

            // Render live scrolling Energy plot (F4)
            if (!paused) {
                appendEnergySample();
            }
            if (showPlotPtr.get(ValueLayout.JAVA_INT, 0) != 0 && vpWidth > 360 && vpHeight > 240) {
                try (Arena temp = Arena.ofConfined()) {
                    MemorySegment figRect = temp.allocate(mjrRect_.layout());
                    int fw = Math.min(340, vpWidth / 2);
                    int fh = Math.min(200, vpHeight / 2);
                    mjrRect_.left(figRect, vpLeft + vpWidth - fw - 12);
                    mjrRect_.bottom(figRect, vpBottom + 12);
                    mjrRect_.width(figRect, fw);
                    mjrRect_.height(figRect, fh);
                    mujoco_h.mjr_figure(figRect, energyFig, context.segment());
                }
            }
        }

        // Refresh dynamic values in Right UI (Control & Joint sliders) and render sidebars
        if (showLeftUi) {
            mujoco_h.mjui_update(-1, -1, ui0, uiState, context.segment());
            mujoco_h.mjui_render(ui0, uiState, context.segment());
        }
        if (showRightUi) {
            mujoco_h.mjui_update(-1, -1, ui1, uiState, context.segment());
            mujoco_h.mjui_render(ui1, uiState, context.segment());
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

    private void initFigure() {
        mujoco_h.mjv_defaultFigure(energyFig);
        mjvFigure_.flg_extend(energyFig, 1);
        mjvFigure_.flg_legend(energyFig, 1);
        setCString(mjvFigure_.title(energyFig), 1000, "Simulation Energy / Trajectory");
        setCString(mjvFigure_.xlabel(energyFig), 100, "Frame");
        MemorySegment grid = mjvFigure_.gridsize(energyFig);
        grid.setAtIndex(ValueLayout.JAVA_INT, 0, 5);
        grid.setAtIndex(ValueLayout.JAVA_INT, 1, 5);

        MemorySegment lineNames = mjvFigure_.linename(energyFig);
        setCString(lineNames.asSlice(0, 100), 100, "Total Z / E");
        setCString(lineNames.asSlice(100, 100), 100, "Kinetic / Vel");

        MemorySegment lineRgb = mjvFigure_.linergb(energyFig);
        // Line 0: Cyan
        lineRgb.setAtIndex(ValueLayout.JAVA_FLOAT, 0, 0.2f);
        lineRgb.setAtIndex(ValueLayout.JAVA_FLOAT, 1, 0.9f);
        lineRgb.setAtIndex(ValueLayout.JAVA_FLOAT, 2, 1.0f);
        // Line 1: Orange
        lineRgb.setAtIndex(ValueLayout.JAVA_FLOAT, 3, 1.0f);
        lineRgb.setAtIndex(ValueLayout.JAVA_FLOAT, 4, 0.6f);
        lineRgb.setAtIndex(ValueLayout.JAVA_FLOAT, 5, 0.2f);
    }

    private void appendEnergySample() {
        MemorySegment linePnt = mjvFigure_.linepnt(energyFig);
        MemorySegment lineData = mjvFigure_.linedata(energyFig);
        int maxPnt = 200;
        int pnt = Math.min(maxPnt, linePnt.getAtIndex(ValueLayout.JAVA_INT, 0) + 1);
        linePnt.setAtIndex(ValueLayout.JAVA_INT, 0, pnt);
        linePnt.setAtIndex(ValueLayout.JAVA_INT, 1, pnt);

        // Shift existing points right
        for (int line = 0; line < 2; line++) {
            MemorySegment row = lineData.asSlice((long) line * 2002 * 4, 2002 * 4);
            for (int i = pnt - 1; i > 0; i--) {
                float prevY = row.getAtIndex(ValueLayout.JAVA_FLOAT, 2 * (i - 1) + 1);
                row.setAtIndex(ValueLayout.JAVA_FLOAT, 2 * i, -i);
                row.setAtIndex(ValueLayout.JAVA_FLOAT, 2 * i + 1, prevY);
            }
        }

        float val0 = model.nq() >= 3 ? (float) data.qpos().get(2) : (model.nq() > 0 ? (float) data.qpos().get(0) : 0f);
        float val1 = model.nv() >= 3 ? (float) data.qvel().get(2) : (model.nv() > 0 ? (float) data.qvel().get(0) : 0f);

        MemorySegment row0 = lineData.asSlice(0, 2002 * 4);
        row0.setAtIndex(ValueLayout.JAVA_FLOAT, 0, 0f);
        row0.setAtIndex(ValueLayout.JAVA_FLOAT, 1, val0);

        MemorySegment row1 = lineData.asSlice(2002 * 4L, 2002 * 4);
        row1.setAtIndex(ValueLayout.JAVA_FLOAT, 0, 0f);
        row1.setAtIndex(ValueLayout.JAVA_FLOAT, 1, val1);
    }

    private void initUiPanels() {
        MemorySegment predStub = mjfItemEnable.allocate((cat, d) -> 1, uiArena);
        MemorySegment themeSpacing = mujoco_h.mjui_themeSpacing(uiArena, 0);
        MemorySegment themeColor = mujoco_h.mjui_themeColor(uiArena, 0); // Dark theme

        // Configure Left UI (ui0)
        mjUI_.spacing(ui0, themeSpacing);
        mjUI_.color(ui0, themeColor);
        mjUI_.predicate(ui0, predStub);
        mjUI_.rectid(ui0, 1);
        mjUI_.auxid(ui0, 0);
        mjUI_.radiocol(ui0, 2);

        // Configure Right UI (ui1)
        mjUI_.spacing(ui1, themeSpacing);
        mjUI_.color(ui1, themeColor);
        mjUI_.predicate(ui1, predStub);
        mjUI_.rectid(ui1, 2);
        mjUI_.auxid(ui1, 1);
        mjUI_.radiocol(ui1, 2);

        buildLeftUiDefinitions();
        buildRightUiDefinitions();
    }

    private void buildLeftUiDefinitions() {
        // Section 0: Simulation
        addDef(ui0, mujoco_h.mjITEM_SECTION(), "Simulation", mujoco_h.mjSECT_OPEN(), MemorySegment.NULL, "");
        addDef(ui0, mujoco_h.mjITEM_RADIOLINE(), "State", 1, runCheckPtr, "Pause\nRun");
        addDef(ui0, mujoco_h.mjITEM_BUTTON(), "Reset (Backspace)", 1, MemorySegment.NULL, "");
        addDef(ui0, mujoco_h.mjITEM_BUTTON(), "Step Once (Right)", 1, MemorySegment.NULL, "");
        addDef(ui0, mujoco_h.mjITEM_BUTTON(), "Align Camera", 1, MemorySegment.NULL, "");
        addDef(ui0, mujoco_h.mjITEM_CHECKINT(), "Info HUD (F2)", 1, showInfoPtr, "");
        addDef(ui0, mujoco_h.mjITEM_CHECKINT(), "Help HUD (F1)", 1, showHelpPtr, "");
        addDef(ui0, mujoco_h.mjITEM_CHECKINT(), "Live Plot (F4)", 1, showPlotPtr, "");

        // Section 1: Rendering (mjvScene flags + camera/label/frame)
        addDef(ui0, mujoco_h.mjITEM_SECTION(), "Rendering", mujoco_h.mjSECT_OPEN(), MemorySegment.NULL, "");
        addDef(ui0, mujoco_h.mjITEM_SELECT(), "Camera", 1,
                camera.segment().asSlice(mjvCamera_.type$offset(), 4), "Free\nTracking");
        addDef(ui0, mujoco_h.mjITEM_SELECT(), "Label", 1,
                option.segment().asSlice(mjvOption_.label$offset(), 4),
                "None\nBody\nJoint\nGeom\nSite\nCamera\nLight\nTendon\nActuator\nConstraint\nFlex\nSkin\nSelection\nSelPoint\nContactForce");
        addDef(ui0, mujoco_h.mjITEM_SELECT(), "Frame", 1,
                option.segment().asSlice(mjvOption_.frame$offset(), 4),
                "None\nBody\nGeom\nSite\nCamera\nLight\nContact\nWorld");
        addDef(ui0, mujoco_h.mjITEM_SEPARATOR(), "Effects", 1, MemorySegment.NULL, "");
        String[] rndNames = {"Shadow", "Wireframe", "Reflection", "Additive", "Skybox", "Fog", "Haze", "Depth", "Segment", "Id Color", "Cull Face"};
        MemorySegment scnFlags = mjvScene_.flags(scene.segment());
        for (int i = 0; i < rndNames.length; i++) {
            addDef(ui0, mujoco_h.mjITEM_CHECKBYTE(), rndNames[i], 1, scnFlags.asSlice(i, 1), "");
        }

        // Section 2: Visualization (mjvOption flags)
        addDef(ui0, mujoco_h.mjITEM_SECTION(), "Visualization", mujoco_h.mjSECT_OPEN(), MemorySegment.NULL, "");
        String[] visNames = {
                "Convex Hull", "Texture", "Joint", "Camera", "Actuator", "Activation",
                "Light", "Tendon", "RangeFinder", "Constraint", "Inertia", "Scale Inertia",
                "Perturb Force", "Perturb Object", "Contact Point", "Contact Force",
                "Contact Split", "Transparent", "Auto Connect", "Center of Mass",
                "Select Point", "Static Body", "Skin"
        };
        MemorySegment visFlags = mjvOption_.flags(option.segment());
        for (int i = 0; i < visNames.length; i++) {
            addDef(ui0, mujoco_h.mjITEM_CHECKBYTE(), visNames[i], 1, visFlags.asSlice(i, 1), "");
        }

        // Section 3: Groups
        addDef(ui0, mujoco_h.mjITEM_SECTION(), "Groups", mujoco_h.mjSECT_CLOSED(), MemorySegment.NULL, "");
        MemorySegment geomGroup = mjvOption_.geomgroup(option.segment());
        MemorySegment siteGroup = mjvOption_.sitegroup(option.segment());
        for (int i = 0; i < 6; i++) {
            addDef(ui0, mujoco_h.mjITEM_CHECKBYTE(), "Geom Group " + i, 1, geomGroup.asSlice(i, 1), "");
        }
        for (int i = 0; i < 6; i++) {
            addDef(ui0, mujoco_h.mjITEM_CHECKBYTE(), "Site Group " + i, 1, siteGroup.asSlice(i, 1), "");
        }
    }

    private void buildRightUiDefinitions() {
        // Section 0: Physics Options
        MemorySegment optSeg = mjModel_.opt(model.segment());
        addDef(ui1, mujoco_h.mjITEM_SECTION(), "Physics", mujoco_h.mjSECT_OPEN(), MemorySegment.NULL, "");
        addDef(ui1, mujoco_h.mjITEM_SELECT(), "Integrator", 1,
                optSeg.asSlice(mjOption_.integrator$offset(), 4), "Euler\nRK4\nImplicit\nImplicitFast");
        MemorySegment gravSeg = mjOption_.gravity(optSeg);
        addDef(ui1, mujoco_h.mjITEM_SLIDERNUM(), "Gravity Z", 1, gravSeg.asSlice(16, 8), "-20 5");

        // Section 1: Actuator Controls
        addDef(ui1, mujoco_h.mjITEM_SECTION(), "Control", mujoco_h.mjSECT_OPEN(), MemorySegment.NULL, "");
        addDef(ui1, mujoco_h.mjITEM_BUTTON(), "Clear Controls", 1, MemorySegment.NULL, "");
        int nu = Math.min(model.nu(), 30);
        if (nu > 0) {
            MemorySegment ctrlPtr = mjData_.ctrl(data.segment()).reinterpret((long) nu * 8);
            for (int i = 0; i < nu; i++) {
                String name = model.id2name(MjObjType.ACTUATOR, i);
                if (name == null || name.isBlank()) {
                    name = "Actuator " + i;
                }
                addDef(ui1, mujoco_h.mjITEM_SLIDERNUM(), truncate(name, 35), 1, ctrlPtr.asSlice((long) i * 8, 8), "-1 1");
            }
        }

        // Section 2: Joint Positions (qpos)
        addDef(ui1, mujoco_h.mjITEM_SECTION(), "Joints (qpos)", mujoco_h.mjSECT_OPEN(), MemorySegment.NULL, "");
        int nq = Math.min(model.nq(), 25);
        if (nq > 0) {
            MemorySegment qposPtr = mjData_.qpos(data.segment()).reinterpret((long) nq * 8);
            for (int i = 0; i < nq; i++) {
                addDef(ui1, mujoco_h.mjITEM_SLIDERNUM(), "qpos[" + i + "]", 1, qposPtr.asSlice((long) i * 8, 8), "-3.14 3.14");
            }
        }
    }

    private void addDef(MemorySegment ui, int type, String name, int state, MemorySegment pdata, String other) {
        try (Arena temp = Arena.ofConfined()) {
            // mjui_add expects an array of mjuiDef terminated by an entry with type == mjITEM_END
            MemorySegment defs = temp.allocate(mjuiDef_.layout(), 2);
            MemorySegment def0 = mjuiDef_.asSlice(defs, 0);
            mjuiDef_.type(def0, type);
            setCString(mjuiDef_.name(def0), 40, name);
            mjuiDef_.state(def0, state);
            mjuiDef_.pdata(def0, pdata);
            setCString(mjuiDef_.other(def0), 300, other);
            mjuiDef_.otherint(def0, 0);

            MemorySegment def1 = mjuiDef_.asSlice(defs, 1);
            mjuiDef_.type(def1, mujoco_h.mjITEM_END());

            mujoco_h.mjui_add(ui, defs);
        }
    }

    private void updateRectanglesAndResizeUi() {
        int[] fb = GlfwBindings.glfwGetFramebufferSize(window);
        int fbW = Math.max(1, fb[0]);
        int fbH = Math.max(1, fb[1]);

        if (fbW == lastFbW && fbH == lastFbH && showLeftUi == lastShowLeft && showRightUi == lastShowRight) {
            return;
        }
        lastFbW = fbW;
        lastFbH = fbH;
        lastShowLeft = showLeftUi;
        lastShowRight = showRightUi;

        mujoco_h.mjui_resize(ui0, context.segment());
        mujoco_h.mjui_resize(ui1, context.segment());

        int w0 = showLeftUi ? mjUI_.width(ui0) : 0;
        int w1 = showRightUi ? mjUI_.width(ui1) : 0;
        int samples = mjuiThemeSpacing_.samples(mjUI_.spacing(ui0));

        if (w0 > 0) {
            mujoco_h.mjr_addAux(0, mjUI_.width(ui0), mjUI_.maxheight(ui0), samples, context.segment());
        }
        if (w1 > 0) {
            mujoco_h.mjr_addAux(1, mjUI_.width(ui1), mjUI_.maxheight(ui1), samples, context.segment());
        }

        mjuiState_.nrect(uiState, 4);
        MemorySegment rects = mjuiState_.rect(uiState);

        // rect[0]: Entire window
        MemorySegment r0 = mjrRect_.asSlice(rects, 0);
        mjrRect_.left(r0, 0);
        mjrRect_.bottom(r0, 0);
        mjrRect_.width(r0, fbW);
        mjrRect_.height(r0, fbH);

        // rect[1]: Left sidebar (ui0)
        MemorySegment r1 = mjrRect_.asSlice(rects, 1);
        mjrRect_.left(r1, 0);
        mjrRect_.bottom(r1, 0);
        mjrRect_.width(r1, w0);
        mjrRect_.height(r1, fbH);

        // rect[2]: Right sidebar (ui1)
        MemorySegment r2 = mjrRect_.asSlice(rects, 2);
        mjrRect_.left(r2, Math.max(w0, fbW - w1));
        mjrRect_.bottom(r2, 0);
        mjrRect_.width(r2, w1);
        mjrRect_.height(r2, fbH);

        // rect[3]: Center 3D Viewport
        MemorySegment r3 = mjrRect_.asSlice(rects, 3);
        mjrRect_.left(r3, w0);
        mjrRect_.bottom(r3, 0);
        mjrRect_.width(r3, Math.max(1, fbW - w0 - w1));
        mjrRect_.height(r3, fbH);
    }

    private void renderHudOverlays(MemorySegment viewport3d) {
        try (Arena temp = Arena.ofConfined()) {
            if (showInfoPtr.get(ValueLayout.JAVA_INT, 0) != 0) {
                int selBody = mjvPerturb_.select(perturb.segment());
                String selName = selBody > 0 ? model.id2name(MjObjType.BODY, selBody) : "None (Double-click object)";
                if (selName == null || selName.isBlank()) {
                    selName = "Body #" + selBody;
                }
                double zPos = model.nq() >= 3 ? data.qpos().get(2) : (model.nq() > 0 ? data.qpos().get(0) : 0.0);

                String leftCol = "Status\nTime\nFPS\nPosition (z)\nContacts\nDOF (nq / nv)\nSelected Body\nUI Panels";
                String rightCol = String.format(
                        "%s\n%.3f s\n%.1f\n%.4f m\n%d\n%d / %d\n%s\nTab (Left) | Shift+Tab (Right)",
                        isPaused() ? "PAUSED (Space)" : "RUNNING (Space)",
                        data.time(),
                        measuredFps,
                        zPos,
                        data.ncon(),
                        model.nq(), model.nv(),
                        selName
                );
                mujoco_h.mjr_overlay(
                        mujoco_h.mjFONT_NORMAL(),
                        mujoco_h.mjGRID_BOTTOMLEFT(),
                        viewport3d,
                        temp.allocateFrom(leftCol),
                        temp.allocateFrom(rightCol),
                        context.segment()
                );
            }

            if (showHelpPtr.get(ValueLayout.JAVA_INT, 0) != 0) {
                String keys = """
                        F1
                        F2
                        F4
                        Tab / Shift+Tab
                        Space
                        Right Arrow
                        Backspace
                        Double-Click Left
                        Double-Click Right
                        Ctrl + Right Drag
                        Ctrl + Left Drag
                        Left / Right Drag
                        Scroll Wheel
                        Esc""";
                String actions = """
                        Toggle Help Overlay
                        Toggle Info HUD
                        Toggle Live Plot
                        Toggle Left / Right Sidebars
                        Pause / Resume Simulation
                        Step Single Frame (when paused)
                        Reset Simulation to t=0
                        Select 3D Body for Perturbation
                        Track Body with Camera
                        Pull / Push Selected Body (3D Force Arrow)
                        Rotate Selected Body (3D Torque)
                        Orbit / Pan Camera
                        Zoom Camera
                        Close Viewer""";
                mujoco_h.mjr_overlay(
                        mujoco_h.mjFONT_NORMAL(),
                        mujoco_h.mjGRID_TOPLEFT(),
                        viewport3d,
                        temp.allocateFrom(keys),
                        temp.allocateFrom(actions),
                        context.segment()
                );
            }
        }
    }

    private void installInputCallbacks() {
        try {
            Linker linker = Linker.nativeLinker();
            MethodHandles.Lookup lookup = MethodHandles.lookup();

            MethodHandle cursorMh = lookup.findVirtual(MjViewer.class, "onCursorPos",
                    MethodType.methodType(void.class, MemorySegment.class, double.class, double.class))
                    .bindTo(this);
            GlfwBindings.glfwSetCursorPosCallback(window,
                    linker.upcallStub(cursorMh, GlfwBindings.CURSOR_POS_CB_DESC, uiArena));

            MethodHandle buttonMh = lookup.findVirtual(MjViewer.class, "onMouseButton",
                    MethodType.methodType(void.class, MemorySegment.class, int.class, int.class, int.class))
                    .bindTo(this);
            GlfwBindings.glfwSetMouseButtonCallback(window,
                    linker.upcallStub(buttonMh, GlfwBindings.MOUSE_BUTTON_CB_DESC, uiArena));

            MethodHandle scrollMh = lookup.findVirtual(MjViewer.class, "onScroll",
                    MethodType.methodType(void.class, MemorySegment.class, double.class, double.class))
                    .bindTo(this);
            GlfwBindings.glfwSetScrollCallback(window,
                    linker.upcallStub(scrollMh, GlfwBindings.SCROLL_CB_DESC, uiArena));

            MethodHandle keyMh = lookup.findVirtual(MjViewer.class, "onKey",
                    MethodType.methodType(void.class, MemorySegment.class, int.class, int.class, int.class, int.class))
                    .bindTo(this);
            GlfwBindings.glfwSetKeyCallback(window,
                    linker.upcallStub(keyMh, GlfwBindings.KEY_CB_DESC, uiArena));
        } catch (ReflectiveOperationException e) {
            throw new MjException("Failed to bind GLFW input callbacks", e);
        }
    }

    private void updateUiModifierKeys() {
        boolean shift = GlfwBindings.glfwGetKey(window, 340) == GlfwBindings.GLFW_PRESS
                || GlfwBindings.glfwGetKey(window, 344) == GlfwBindings.GLFW_PRESS;
        boolean ctrl = GlfwBindings.glfwGetKey(window, 341) == GlfwBindings.GLFW_PRESS
                || GlfwBindings.glfwGetKey(window, 345) == GlfwBindings.GLFW_PRESS;
        boolean alt = GlfwBindings.glfwGetKey(window, 342) == GlfwBindings.GLFW_PRESS
                || GlfwBindings.glfwGetKey(window, 346) == GlfwBindings.GLFW_PRESS;
        mjuiState_.shift(uiState, shift ? 1 : 0);
        mjuiState_.control(uiState, ctrl ? 1 : 0);
        mjuiState_.alt(uiState, alt ? 1 : 0);
    }

    private void updateMouseRect(double glX, double glY) {
        MemorySegment rects = mjuiState_.rect(uiState);
        int found = mujoco_h.mjr_findRect((int) Math.round(glX), (int) Math.round(glY), 3, mjrRect_.asSlice(rects, 1));
        mjuiState_.mouserect(uiState, found + 1);
    }

    private void dispatchUiEvent(int eventType) {
        mjuiState_.type(uiState, eventType);
        if (showLeftUi) {
            MemorySegment item = mujoco_h.mjui_event(ui0, uiState, context.segment());
            if (item != null && !item.equals(MemorySegment.NULL)) {
                MemorySegment it = item.reinterpret(mjuiItem_.sizeof());
                handleLeftUiAction(mjuiItem_.sectionid(it), mjuiItem_.itemid(it));
            }
        }
        if (showRightUi) {
            MemorySegment item = mujoco_h.mjui_event(ui1, uiState, context.segment());
            if (item != null && !item.equals(MemorySegment.NULL)) {
                MemorySegment it = item.reinterpret(mjuiItem_.sizeof());
                handleRightUiAction(mjuiItem_.sectionid(it), mjuiItem_.itemid(it));
            }
        }
    }

    private void handleLeftUiAction(int sectionId, int itemId) {
        if (sectionId == 0) {
            if (itemId == 1) { // Reset
                MjSim.resetData(model, data);
                MjSim.forward(model, data);
                if (isPaused()) {
                    MjSim.getState(model, data, pausedStateSnapshot, MjState.INTEGRATION.value());
                }
            } else if (itemId == 2) { // Step Once
                stepOnceRequested = true;
            } else if (itemId == 3) { // Align Camera
                mujoco_h.mjv_defaultFreeCamera(model.segment(), camera.segment());
            }
        }
    }

    private void handleRightUiAction(int sectionId, int itemId) {
        if (sectionId == 1 && itemId == 0) { // Clear Controls
            data.ctrl().fill(0.0);
        }
    }

    private void onCursorPos(MemorySegment win, double xpos, double ypos) {
        int[] fb = GlfwBindings.glfwGetFramebufferSize(window);
        int[] ws = GlfwBindings.glfwGetWindowSize(window);
        double scaleX = ws[0] > 0 ? (double) fb[0] / ws[0] : 1.0;
        double scaleY = ws[1] > 0 ? (double) fb[1] / ws[1] : 1.0;

        double glX = xpos * scaleX;
        double glY = fb[1] - ypos * scaleY;

        double dx = glX - mjuiState_.x(uiState);
        double dy = glY - mjuiState_.y(uiState);
        mjuiState_.x(uiState, glX);
        mjuiState_.y(uiState, glY);
        mjuiState_.dx(uiState, dx);
        mjuiState_.dy(uiState, dy);

        updateUiModifierKeys();
        updateMouseRect(glX, glY);

        int dragRect = mjuiState_.dragrect(uiState);
        if (dragRect == 1 || dragRect == 2 || mjuiState_.mouserect(uiState) == 1 || mjuiState_.mouserect(uiState) == 2) {
            dispatchUiEvent(mujoco_h.mjEVENT_MOVE());
            if (dragRect == 1 || dragRect == 2) {
                return;
            }
        }

        boolean left = mjuiState_.left(uiState) != 0;
        boolean right = mjuiState_.right(uiState) != 0;
        boolean middle = mjuiState_.middle(uiState) != 0;
        if (!left && !right && !middle) {
            return;
        }

        int height = Math.max(1, fb[1]);
        double relDx = dx / height;
        double relDy = -dy / height;

        boolean ctrl = mjuiState_.control(uiState) != 0;
        boolean shift = mjuiState_.shift(uiState) != 0;

        // If Ctrl is held and a body is selected, move 3D perturbation force/pose!
        if (ctrl && mjvPerturb_.select(perturb.segment()) > 0) {
            int action = right
                    ? (shift ? mujoco_h.mjMOUSE_MOVE_H() : mujoco_h.mjMOUSE_MOVE_V())
                    : (shift ? mujoco_h.mjMOUSE_ROTATE_H() : mujoco_h.mjMOUSE_ROTATE_V());
            mujoco_h.mjv_movePerturb(model.segment(), data.segment(), action, relDx, relDy, scene.segment(), perturb.segment());
            return;
        }

        int action;
        if (right) {
            action = shift ? mujoco_h.mjMOUSE_MOVE_H() : mujoco_h.mjMOUSE_MOVE_V();
        } else if (left) {
            action = shift ? mujoco_h.mjMOUSE_ROTATE_H() : mujoco_h.mjMOUSE_ROTATE_V();
        } else {
            action = mujoco_h.mjMOUSE_ZOOM();
        }
        if (model.isOpen()) {
            mujoco_h.mjv_moveCamera(model.segment(), action, relDx, relDy, camera.segment());
        }
    }

    private void onMouseButton(MemorySegment win, int button, int act, int mods) {
        updateUiModifierKeys();
        boolean pressed = (act == GlfwBindings.GLFW_PRESS);

        int mjBtn = switch (button) {
            case GlfwBindings.GLFW_MOUSE_BUTTON_LEFT -> mujoco_h.mjBUTTON_LEFT();
            case GlfwBindings.GLFW_MOUSE_BUTTON_RIGHT -> mujoco_h.mjBUTTON_RIGHT();
            case GlfwBindings.GLFW_MOUSE_BUTTON_MIDDLE -> mujoco_h.mjBUTTON_MIDDLE();
            default -> mujoco_h.mjBUTTON_NONE();
        };

        if (button == GlfwBindings.GLFW_MOUSE_BUTTON_LEFT) {
            mjuiState_.left(uiState, pressed ? 1 : 0);
        } else if (button == GlfwBindings.GLFW_MOUSE_BUTTON_RIGHT) {
            mjuiState_.right(uiState, pressed ? 1 : 0);
        } else if (button == GlfwBindings.GLFW_MOUSE_BUTTON_MIDDLE) {
            mjuiState_.middle(uiState, pressed ? 1 : 0);
        }

        mjuiState_.button(uiState, mjBtn);
        double nowSec = System.nanoTime() / 1_000_000_000.0;
        boolean isDoubleClick = pressed && (mjBtn == lastClickButton) && ((nowSec - lastClickTimeSec) < 0.28);
        mjuiState_.doubleclick(uiState, isDoubleClick ? 1 : 0);
        if (pressed) {
            lastClickTimeSec = nowSec;
            lastClickButton = mjBtn;
            mjuiState_.buttontime(uiState, nowSec);
            mjuiState_.dragrect(uiState, mjuiState_.mouserect(uiState));
            mjuiState_.dragbutton(uiState, mjBtn);
        } else {
            mjuiState_.dragrect(uiState, 0);
            mjvPerturb_.active(perturb.segment(), 0);
        }

        int mouseRect = mjuiState_.mouserect(uiState);
        if (mouseRect == 1 || mouseRect == 2) {
            dispatchUiEvent(pressed ? mujoco_h.mjEVENT_PRESS() : mujoco_h.mjEVENT_RELEASE());
            return;
        }

        // 3D Viewport double-click body selection & Ctrl+Drag perturbation activation
        if (pressed && mouseRect == 3) {
            if (isDoubleClick) {
                select3dBody(mjBtn);
            } else if (mjuiState_.control(uiState) != 0 && mjvPerturb_.select(perturb.segment()) > 0) {
                int newPert = (mjBtn == mujoco_h.mjBUTTON_RIGHT()) ? mujoco_h.mjPERT_TRANSLATE() : mujoco_h.mjPERT_ROTATE();
                mjvPerturb_.active(perturb.segment(), newPert);
                mujoco_h.mjv_initPerturb(model.segment(), data.segment(), scene.segment(), perturb.segment());
            }
        }
    }

    private void select3dBody(int mjBtn) {
        MemorySegment rects = mjuiState_.rect(uiState);
        MemorySegment vp = mjrRect_.asSlice(rects, 3);
        int vpW = Math.max(1, mjrRect_.width(vp));
        int vpH = Math.max(1, mjrRect_.height(vp));
        double relX = (mjuiState_.x(uiState) - mjrRect_.left(vp)) / vpW;
        double relY = (mjuiState_.y(uiState) - mjrRect_.bottom(vp)) / vpH;
        double aspect = (double) vpW / vpH;

        try (Arena temp = Arena.ofConfined()) {
            MemorySegment selPnt = temp.allocate(ValueLayout.JAVA_DOUBLE, 3);
            MemorySegment geomId = temp.allocateFrom(ValueLayout.JAVA_INT, -1);
            MemorySegment flexId = temp.allocateFrom(ValueLayout.JAVA_INT, -1);
            MemorySegment skinId = temp.allocateFrom(ValueLayout.JAVA_INT, -1);

            int bodyId = mujoco_h.mjv_select(
                    model.segment(), data.segment(), option.segment(),
                    aspect, relX, relY, scene.segment(),
                    selPnt, geomId, flexId, skinId
            );

            if (mjBtn == mujoco_h.mjBUTTON_LEFT()) {
                if (bodyId > 0) {
                    mjvPerturb_.select(perturb.segment(), bodyId);
                    mjvPerturb_.flexselect(perturb.segment(), flexId.get(ValueLayout.JAVA_INT, 0));
                    mjvPerturb_.skinselect(perturb.segment(), skinId.get(ValueLayout.JAVA_INT, 0));
                    mujoco_h.mjv_initPerturb(model.segment(), data.segment(), scene.segment(), perturb.segment());
                } else {
                    mjvPerturb_.select(perturb.segment(), 0);
                }
            } else if (mjBtn == mujoco_h.mjBUTTON_RIGHT()) {
                if (bodyId > 0) {
                    mjvCamera_.type(camera.segment(), mujoco_h.mjCAMERA_TRACKING());
                    mjvCamera_.trackbodyid(camera.segment(), bodyId);
                } else {
                    mjvCamera_.type(camera.segment(), mujoco_h.mjCAMERA_FREE());
                }
            }
        }
    }

    private void onScroll(MemorySegment win, double xoffset, double yoffset) {
        updateUiModifierKeys();
        mjuiState_.sx(uiState, xoffset);
        mjuiState_.sy(uiState, yoffset);

        int mouseRect = mjuiState_.mouserect(uiState);
        if (mouseRect == 1 || mouseRect == 2) {
            dispatchUiEvent(mujoco_h.mjEVENT_SCROLL());
            return;
        }
        if (model.isOpen()) {
            mujoco_h.mjv_moveCamera(model.segment(), mujoco_h.mjMOUSE_ZOOM(), 0.0, -0.05 * yoffset, camera.segment());
        }
    }

    private void onKey(MemorySegment win, int key, int scancode, int act, int mods) {
        if (act != GlfwBindings.GLFW_PRESS) {
            return;
        }
        updateUiModifierKeys();
        mjuiState_.key(uiState, key);
        mjuiState_.keytime(uiState, System.nanoTime() / 1_000_000_000.0);

        boolean shift = (mods & GlfwBindings.GLFW_MOD_SHIFT) != 0;

        if (key == GlfwBindings.GLFW_KEY_ESCAPE) {
            GlfwBindings.glfwSetWindowShouldClose(window, true);
        } else if (key == GlfwBindings.GLFW_KEY_SPACE) {
            setPaused(!isPaused());
        } else if (key == KEY_RIGHT && isPaused()) {
            stepOnceRequested = true;
        } else if (key == GlfwBindings.GLFW_KEY_BACKSPACE) {
            MjSim.resetData(model, data);
            MjSim.forward(model, data);
            if (isPaused()) {
                MjSim.getState(model, data, pausedStateSnapshot, MjState.INTEGRATION.value());
            }
        } else if (key == KEY_TAB) {
            if (shift) {
                showRightUi = !showRightUi;
            } else {
                showLeftUi = !showLeftUi;
            }
        } else if (key == KEY_F1) {
            showHelpPtr.set(ValueLayout.JAVA_INT, 0, showHelpPtr.get(ValueLayout.JAVA_INT, 0) == 0 ? 1 : 0);
        } else if (key == KEY_F2) {
            showInfoPtr.set(ValueLayout.JAVA_INT, 0, showInfoPtr.get(ValueLayout.JAVA_INT, 0) == 0 ? 1 : 0);
        } else if (key == KEY_F4) {
            showPlotPtr.set(ValueLayout.JAVA_INT, 0, showPlotPtr.get(ValueLayout.JAVA_INT, 0) == 0 ? 1 : 0);
        } else {
            dispatchUiEvent(mujoco_h.mjEVENT_KEY());
        }
    }

    private static void setCString(MemorySegment target, int maxBytes, String value) {
        target.fill((byte) 0);
        if (value == null || value.isEmpty()) {
            return;
        }
        byte[] bytes = value.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        int len = Math.min(bytes.length, maxBytes - 1);
        MemorySegment.copy(bytes, 0, target, ValueLayout.JAVA_BYTE, 0, len);
    }

    private static String truncate(String s, int maxLen) {
        return s.length() <= maxLen ? s : s.substring(0, maxLen);
    }

    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            try {
                GlfwBindings.glfwMakeContextCurrent(window);
                context.close();
                scene.close();
                GlfwBindings.glfwDestroyWindow(window);
                uiArena.close();
            } finally {
                GlfwBindings.glfwTerminate();
            }
        }
    }
}
