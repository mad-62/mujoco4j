package org.mujoco4j;

import org.mujoco4j.internal.panama.mjWarningStat_;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.function.BooleanSupplier;

/**
 * Wrapper around {@code mjWarningStat}.
 */
public final class MjWarningStat {

    private final MemorySegment segment;
    private final BooleanSupplier livenessGuard;

    public MjWarningStat() {
        MjNativeLoader.ensureLoaded();
        this.segment = Arena.ofAuto().allocate(mjWarningStat_.layout());
        this.livenessGuard = () -> true;
    }

    public MjWarningStat(MemorySegment segment, BooleanSupplier livenessGuard) {
        this.segment = segment;
        this.livenessGuard = livenessGuard != null ? livenessGuard : () -> true;
    }

    private void checkOpen() {
        if (!livenessGuard.getAsBoolean() || !segment.scope().isAlive()) {
            throw new IllegalStateException("Underlying MjWarningStat memory has already been closed.");
        }
    }

    public int lastinfo() {
        checkOpen();
        return mjWarningStat_.lastinfo(segment);
    }

    public void setLastinfo(int lastinfo) {
        checkOpen();
        mjWarningStat_.lastinfo(segment, lastinfo);
    }

    public int number() {
        checkOpen();
        return mjWarningStat_.number(segment);
    }

    public void setNumber(int number) {
        checkOpen();
        mjWarningStat_.number(segment, number);
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof MjWarningStat other)) return false;
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
        return "<MjWarningStat lastinfo: " + lastinfo() + ", number: " + number() + ">";
    }
}
