package org.mujoco4j;

import org.mujoco4j.internal.panama.mjvGeom_;

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

    public int type() {
        if (!livenessGuard.getAsBoolean()) throw new IllegalStateException("Closed");
        return mjvGeom_.type(segment);
    }

    public MemorySegment segment() {
        return segment;
    }
}
