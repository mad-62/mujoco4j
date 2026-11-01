package org.mujoco4j;

import org.mujoco4j.internal.panama.mjrContext_;
import org.mujoco4j.internal.panama.mjrRect_;
import org.mujoco4j.internal.panama.mujoco_h;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Safe {@link AutoCloseable} wrapper around MuJoCo's OpenGL {@code mjrContext}.
 * Requires an active OpenGL context on the calling thread when created, rendered, or closed.
 */
public final class MjrContext implements AutoCloseable {

    public static final int FONTSCALE_50 = mujoco_h.mjFONTSCALE_50();
    public static final int FONTSCALE_100 = mujoco_h.mjFONTSCALE_100();
    public static final int FONTSCALE_150 = mujoco_h.mjFONTSCALE_150();
    public static final int FONTSCALE_200 = mujoco_h.mjFONTSCALE_200();

    public static final int FB_WINDOW = mujoco_h.mjFB_WINDOW();
    public static final int FB_OFFSCREEN = mujoco_h.mjFB_OFFSCREEN();

    private final Arena arena;
    private final MemorySegment segment;
    private final MemorySegment viewportSeg;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    public MjrContext(MjModel model, int fontScale) {
        MjNativeLoader.ensureLoaded();
        if (model != null) {
            model.checkOpen();
        }
        this.arena = Arena.ofShared();
        MemorySegment raw = arena.allocate(mjrContext_.layout());
        mujoco_h.mjr_defaultContext(raw);
        mujoco_h.mjr_makeContext(model != null ? model.segment() : MemorySegment.NULL, raw, fontScale);
        this.segment = raw.reinterpret(mjrContext_.sizeof(), arena, mujoco_h::mjr_freeContext);
        this.viewportSeg = arena.allocate(mjrRect_.layout());
    }

    public MjrContext(MjModel model) {
        this(model, FONTSCALE_150);
    }

    public boolean isOpen() {
        return !closed.get() && arena.scope().isAlive();
    }

    public void checkOpen() {
        if (!isOpen()) {
            throw new IllegalStateException("MjrContext has already been closed.");
        }
    }

    public void setBuffer(int framebuffer) {
        checkOpen();
        mujoco_h.mjr_setBuffer(framebuffer, segment);
    }

    public void render(int left, int bottom, int width, int height, MjvScene scene) {
        checkOpen();
        scene.checkOpen();
        mjrRect_.left(viewportSeg, left);
        mjrRect_.bottom(viewportSeg, bottom);
        mjrRect_.width(viewportSeg, width);
        mjrRect_.height(viewportSeg, height);
        mujoco_h.mjr_render(viewportSeg, scene.segment(), segment);
    }

    public MemorySegment segment() {
        checkOpen();
        return segment;
    }

    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            arena.close();
        }
    }
}
