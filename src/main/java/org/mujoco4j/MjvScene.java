package org.mujoco4j;

import org.mujoco4j.internal.panama.mjvGeom_;
import org.mujoco4j.internal.panama.mjvScene_;
import org.mujoco4j.internal.panama.mujoco_h;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Safe {@link AutoCloseable} wrapper around {@code mjvScene}.
 */
public final class MjvScene implements AutoCloseable {

    private final Arena arena;
    private final MemorySegment segment;
    private final AtomicBoolean closed = new AtomicBoolean(false);
    private final List<MjvGeom> geomsView;

    public MjvScene() {
        this(null, 0);
    }

    public MjvScene(MjModel model, int maxgeom) {
        MjNativeLoader.ensureLoaded();
        if (maxgeom < 0) {
            throw new IllegalArgumentException("maxgeom must be >= 0");
        }
        this.arena = Arena.ofShared();
        MemorySegment raw = arena.allocate(mjvScene_.layout());
        mujoco_h.mjv_defaultScene(raw);
        mujoco_h.mjv_makeScene(model != null ? model.segment() : MemorySegment.NULL, raw, maxgeom);
        this.segment = raw.reinterpret(mjvScene_.sizeof(), arena, mujoco_h::mjv_freeScene);

        int count = mjvScene_.maxgeom(this.segment);
        List<MjvGeom> list = new ArrayList<>(count);
        if (count > 0) {
            MemorySegment geomsPtr = mjvScene_.geoms(this.segment).reinterpret((long) count * mjvGeom_.sizeof(), arena, null);
            for (int i = 0; i < count; i++) {
                list.add(new MjvGeom(mjvGeom_.asSlice(geomsPtr, i), this::isOpen));
            }
        }
        this.geomsView = Collections.unmodifiableList(list);
    }

    public boolean isOpen() {
        return !closed.get() && arena.scope().isAlive();
    }

    public void checkOpen() {
        if (!isOpen()) {
            throw new IllegalStateException("MjvScene has already been closed.");
        }
    }

    public int maxgeom() {
        checkOpen();
        return mjvScene_.maxgeom(segment);
    }

    public int ngeom() {
        checkOpen();
        return mjvScene_.ngeom(segment);
    }

    public float scale() {
        checkOpen();
        return mjvScene_.scale(segment);
    }

    public List<MjvGeom> geoms() {
        checkOpen();
        return geomsView;
    }

    public void updateScene(MjModel model, MjData data, MjvOption opt, MjvPerturb pert, MjvCamera cam, int catmask) {
        checkOpen();
        model.checkOpen();
        data.checkOpen();
        mujoco_h.mjv_updateScene(
                model.segment(),
                data.segment(),
                opt != null ? opt.segment() : new MjvOption().segment(),
                pert != null ? pert.segment() : MemorySegment.NULL,
                cam != null ? cam.segment() : new MjvCamera().segment(),
                catmask,
                this.segment
        );
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
