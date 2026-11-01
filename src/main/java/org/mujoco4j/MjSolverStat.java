package org.mujoco4j;

import org.mujoco4j.internal.panama.mjSolverStat_;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.function.BooleanSupplier;

/**
 * Wrapper around {@code mjSolverStat}.
 */
public final class MjSolverStat {

    private final MemorySegment segment;
    private final BooleanSupplier livenessGuard;

    public MjSolverStat() {
        MjNativeLoader.ensureLoaded();
        this.segment = Arena.ofAuto().allocate(mjSolverStat_.layout());
        this.livenessGuard = () -> true;
    }

    public MjSolverStat(MemorySegment segment, BooleanSupplier livenessGuard) {
        this.segment = segment;
        this.livenessGuard = livenessGuard != null ? livenessGuard : () -> true;
    }

    private void checkOpen() {
        if (!livenessGuard.getAsBoolean() || !segment.scope().isAlive()) {
            throw new IllegalStateException("Underlying MjSolverStat memory has already been closed.");
        }
    }

    public int neval() {
        checkOpen();
        return mjSolverStat_.neval(segment);
    }

    public void setNeval(int neval) {
        checkOpen();
        mjSolverStat_.neval(segment, neval);
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof MjSolverStat other)) return false;
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
        return "<MjSolverStat neval: " + neval() + ">";
    }
}
