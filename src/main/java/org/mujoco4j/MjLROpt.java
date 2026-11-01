package org.mujoco4j;

import org.mujoco4j.internal.panama.mjLROpt_;
import org.mujoco4j.internal.panama.mujoco_h;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.function.BooleanSupplier;

/**
 * Wrapper around {@code mjLROpt}.
 */
public final class MjLROpt {

    private final MemorySegment segment;
    private final BooleanSupplier livenessGuard;

    public MjLROpt() {
        MjNativeLoader.ensureLoaded();
        this.segment = Arena.ofAuto().allocate(mjLROpt_.layout());
        this.livenessGuard = () -> true;
        mujoco_h.mj_defaultLROpt(this.segment);
    }

    public MjLROpt(MemorySegment segment, BooleanSupplier livenessGuard) {
        this.segment = segment;
        this.livenessGuard = livenessGuard != null ? livenessGuard : () -> true;
    }

    private void checkOpen() {
        if (!livenessGuard.getAsBoolean() || !segment.scope().isAlive()) {
            throw new IllegalStateException("Underlying MjLROpt memory has already been closed.");
        }
    }

    MemorySegment segment() {
        checkOpen();
        return segment;
    }

    public MjLROpt copy() {
        checkOpen();
        MjLROpt dst = new MjLROpt();
        MemorySegment.copy(this.segment, 0, dst.segment, 0, mjLROpt_.sizeof());
        return dst;
    }

    public double maxforce() {
        checkOpen();
        return mjLROpt_.maxforce(segment);
    }

    public void setMaxforce(double maxforce) {
        checkOpen();
        mjLROpt_.maxforce(segment, maxforce);
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof MjLROpt other)) return false;
        checkOpen();
        other.checkOpen();
        return segment.mismatch(other.segment) == -1L;
    }

    @Override
    public int hashCode() {
        throw new UnsupportedOperationException("Mutable MuJoCo structs do not support hashCode()");
    }

    @Override
    public String toString() {
        checkOpen();
        return "<MjLROpt maxforce: " + maxforce() + ">";
    }
}
