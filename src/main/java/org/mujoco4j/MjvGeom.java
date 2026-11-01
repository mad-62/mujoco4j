package org.mujoco4j;

import org.mujoco4j.internal.panama.mjvGeom_;
import org.mujoco4j.internal.panama.mujoco_h;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.function.BooleanSupplier;

/**
 * Wrapper around {@code mjvGeom}.
 */
public final class MjvGeom {

    private final MemorySegment segment;
    private final BooleanSupplier livenessGuard;

    public MjvGeom() {
        MjNativeLoader.ensureLoaded();
        this.segment = Arena.ofAuto().allocate(mjvGeom_.layout());
        this.livenessGuard = () -> true;
    }

    public MjvGeom(MemorySegment segment, BooleanSupplier livenessGuard) {
        this.segment = segment;
        this.livenessGuard = livenessGuard != null ? livenessGuard : () -> true;
    }

    private void checkOpen() {
        if (!livenessGuard.getAsBoolean()) {
            throw new IllegalStateException("Parent MjvScene has been closed.");
        }
    }

    public int type() {
        checkOpen();
        return mjvGeom_.type(segment);
    }

    public int objtype() {
        checkOpen();
        return mjvGeom_.objtype(segment);
    }

    public int objid() {
        checkOpen();
        return mjvGeom_.objid(segment);
    }

    public int category() {
        checkOpen();
        return mjvGeom_.category(segment);
    }

    public float[] size() {
        checkOpen();
        MemorySegment s = mjvGeom_.size(segment);
        return new float[]{
                s.getAtIndex(mujoco_h.C_FLOAT, 0),
                s.getAtIndex(mujoco_h.C_FLOAT, 1),
                s.getAtIndex(mujoco_h.C_FLOAT, 2)
        };
    }

    public float[] pos() {
        checkOpen();
        MemorySegment p = mjvGeom_.pos(segment);
        return new float[]{
                p.getAtIndex(mujoco_h.C_FLOAT, 0),
                p.getAtIndex(mujoco_h.C_FLOAT, 1),
                p.getAtIndex(mujoco_h.C_FLOAT, 2)
        };
    }

    public float[] mat() {
        checkOpen();
        MemorySegment m = mjvGeom_.mat(segment);
        float[] res = new float[9];
        for (int i = 0; i < 9; i++) {
            res[i] = m.getAtIndex(mujoco_h.C_FLOAT, i);
        }
        return res;
    }

    public float[] rgba() {
        checkOpen();
        MemorySegment c = mjvGeom_.rgba(segment);
        return new float[]{
                c.getAtIndex(mujoco_h.C_FLOAT, 0),
                c.getAtIndex(mujoco_h.C_FLOAT, 1),
                c.getAtIndex(mujoco_h.C_FLOAT, 2),
                c.getAtIndex(mujoco_h.C_FLOAT, 3)
        };
    }

    public MemorySegment segment() {
        checkOpen();
        return segment;
    }
}
