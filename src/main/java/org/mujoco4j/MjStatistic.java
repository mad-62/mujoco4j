package org.mujoco4j;

import org.mujoco4j.internal.panama.mjStatistic_;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.function.BooleanSupplier;

/**
 * Wrapper around {@code mjStatistic} (model statistics), either as a live view into {@link MjModel}
 * or as a standalone value object.
 */
public final class MjStatistic {

    private final MemorySegment segment;
    private final BooleanSupplier livenessGuard;
    private final MjDoubleVector centerView;

    public MjStatistic() {
        MjNativeLoader.ensureLoaded();
        Arena arena = Arena.ofAuto();
        this.segment = arena.allocate(mjStatistic_.layout());
        this.livenessGuard = () -> true;
        this.centerView = new MjDoubleVector(mjStatistic_.center(segment), 3, livenessGuard);
    }

    public MjStatistic(MemorySegment segment, BooleanSupplier livenessGuard) {
        this.segment = segment;
        this.livenessGuard = livenessGuard != null ? livenessGuard : () -> true;
        this.centerView = new MjDoubleVector(mjStatistic_.center(segment), 3, this.livenessGuard);
    }

    private void checkOpen() {
        if (!livenessGuard.getAsBoolean() || !segment.scope().isAlive()) {
            throw new IllegalStateException("Underlying MjStatistic memory has already been closed.");
        }
    }

    public MjStatistic copy() {
        checkOpen();
        MjStatistic copy = new MjStatistic();
        copy.copyFrom(this);
        return copy;
    }

    public void copyFrom(MjStatistic other) {
        checkOpen();
        other.checkOpen();
        MemorySegment.copy(other.segment, 0, this.segment, 0, mjStatistic_.sizeof());
    }

    public double meaninertia() {
        checkOpen();
        return mjStatistic_.meaninertia(segment);
    }

    public void setMeaninertia(double value) {
        checkOpen();
        mjStatistic_.meaninertia(segment, value);
    }

    public double meanmass() {
        checkOpen();
        return mjStatistic_.meanmass(segment);
    }

    public void setMeanmass(double value) {
        checkOpen();
        mjStatistic_.meanmass(segment, value);
    }

    public double meansize() {
        checkOpen();
        return mjStatistic_.meansize(segment);
    }

    public void setMeansize(double value) {
        checkOpen();
        mjStatistic_.meansize(segment, value);
    }

    public double extent() {
        checkOpen();
        return mjStatistic_.extent(segment);
    }

    public void setExtent(double value) {
        checkOpen();
        mjStatistic_.extent(segment, value);
    }

    public MjDoubleVector center() {
        checkOpen();
        return centerView;
    }

    public void setCenter(double x, double y, double z) {
        checkOpen();
        centerView.set(0, x);
        centerView.set(1, y);
        centerView.set(2, z);
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof MjStatistic other)) return false;
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
        return "<MjStatistic extent: " + extent() + ", meanmass: " + meanmass() + ", center: " + center() + ">";
    }
}
