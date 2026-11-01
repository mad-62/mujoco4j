package org.mujoco4j;

import org.mujoco4j.internal.panama.mjContact_;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.function.BooleanSupplier;

/**
 * Wrapper around {@code mjContact}, either as a live view into {@link MjData#contacts()}
 * or as a standalone value object.
 */
public final class MjContact {

    private final MemorySegment segment;
    private final BooleanSupplier livenessGuard;
    private final MjDoubleVector posView;
    private final MjDoubleVector frameView;
    private final MjDoubleVector frictionView;
    private final MjDoubleVector hView;
    private final MjIntVector geomView;

    public MjContact() {
        MjNativeLoader.ensureLoaded();
        this.segment = Arena.ofAuto().allocate(mjContact_.layout());
        this.livenessGuard = () -> true;
        this.posView = new MjDoubleVector(mjContact_.pos(segment), 3, livenessGuard);
        this.frameView = new MjDoubleVector(mjContact_.frame(segment), 9, livenessGuard);
        this.frictionView = new MjDoubleVector(mjContact_.friction(segment), 5, livenessGuard);
        this.hView = new MjDoubleVector(mjContact_.H(segment), 36, livenessGuard);
        this.geomView = new MjIntVector(mjContact_.geom(segment), 2, livenessGuard);
    }

    public MjContact(MemorySegment segment, BooleanSupplier livenessGuard) {
        this.segment = segment;
        this.livenessGuard = livenessGuard != null ? livenessGuard : () -> true;
        this.posView = new MjDoubleVector(mjContact_.pos(segment), 3, this.livenessGuard);
        this.frameView = new MjDoubleVector(mjContact_.frame(segment), 9, this.livenessGuard);
        this.frictionView = new MjDoubleVector(mjContact_.friction(segment), 5, this.livenessGuard);
        this.hView = new MjDoubleVector(mjContact_.H(segment), 36, this.livenessGuard);
        this.geomView = new MjIntVector(mjContact_.geom(segment), 2, this.livenessGuard);
    }

    private void checkOpen() {
        if (!livenessGuard.getAsBoolean() || !segment.scope().isAlive()) {
            throw new IllegalStateException("Underlying MjContact memory has already been closed.");
        }
    }

    public MjContact copy() {
        checkOpen();
        MjContact copy = new MjContact();
        MemorySegment.copy(this.segment, 0, copy.segment, 0, mjContact_.sizeof());
        return copy;
    }

    public double dist() {
        checkOpen();
        return mjContact_.dist(segment);
    }

    public void setDist(double dist) {
        checkOpen();
        mjContact_.dist(segment, dist);
    }

    public MjDoubleVector pos() {
        checkOpen();
        return posView;
    }

    public MjDoubleVector frame() {
        checkOpen();
        return frameView;
    }

    public MjDoubleVector friction() {
        checkOpen();
        return frictionView;
    }

    public MjDoubleVector H() {
        checkOpen();
        return hView;
    }

    public MjIntVector geom() {
        checkOpen();
        return geomView;
    }

    public int geom1() {
        checkOpen();
        return geomView.get(0);
    }

    public int geom2() {
        checkOpen();
        return geomView.get(1);
    }

    public int dim() {
        checkOpen();
        return mjContact_.dim(segment);
    }

    public int efcAddress() {
        checkOpen();
        return mjContact_.efc_address(segment);
    }

    public MemorySegment segment() {
        checkOpen();
        return segment;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof MjContact other)) return false;
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
        return "<MjContact dist: " + dist() + ", pos: " + pos() + ", geom: " + geom() + ">";
    }
}
