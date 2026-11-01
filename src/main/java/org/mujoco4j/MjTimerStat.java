package org.mujoco4j;

import org.mujoco4j.internal.panama.mjTimerStat_;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.function.BooleanSupplier;

/**
 * Wrapper around {@code mjTimerStat}.
 */
public final class MjTimerStat {

    private final MemorySegment segment;
    private final BooleanSupplier livenessGuard;

    public MjTimerStat() {
        MjNativeLoader.ensureLoaded();
        this.segment = Arena.ofAuto().allocate(mjTimerStat_.layout());
        this.livenessGuard = () -> true;
    }

    public MjTimerStat(MemorySegment segment, BooleanSupplier livenessGuard) {
        this.segment = segment;
        this.livenessGuard = livenessGuard != null ? livenessGuard : () -> true;
    }

    private void checkOpen() {
        if (!livenessGuard.getAsBoolean() || !segment.scope().isAlive()) {
            throw new IllegalStateException("Underlying MjTimerStat memory has already been closed.");
        }
    }

    public double duration() {
        checkOpen();
        return mjTimerStat_.duration(segment);
    }

    public void setDuration(double duration) {
        checkOpen();
        mjTimerStat_.duration(segment, duration);
    }

    public int number() {
        checkOpen();
        return mjTimerStat_.number(segment);
    }

    public void setNumber(int number) {
        checkOpen();
        mjTimerStat_.number(segment, number);
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof MjTimerStat other)) return false;
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
        return "<MjTimerStat duration: " + duration() + ", number: " + number() + ">";
    }
}
