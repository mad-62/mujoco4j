package org.mujoco4j;

import org.mujoco4j.internal.panama.mjVisual_;
import org.mujoco4j.internal.panama.mujoco_h;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.function.BooleanSupplier;

/**
 * Wrapper around {@code mjVisual} (visualization options) and its nested sub-structs.
 */
public final class MjVisual {

    private final MemorySegment segment;
    private final BooleanSupplier livenessGuard;
    private final Global globalView;
    private final Quality qualityView;
    private final Headlight headlightView;
    private final Map mapView;
    private final Scale scaleView;
    private final Rgba rgbaView;

    public MjVisual() {
        MjNativeLoader.ensureLoaded();
        Arena arena = Arena.ofAuto();
        this.segment = arena.allocate(mjVisual_.layout());
        this.livenessGuard = () -> true;
        mujoco_h.mj_defaultVisual(this.segment);
        this.globalView = new Global(mjVisual_.global(segment), livenessGuard);
        this.qualityView = new Quality(mjVisual_.quality(segment), livenessGuard);
        this.headlightView = new Headlight(mjVisual_.headlight(segment), livenessGuard);
        this.mapView = new Map(mjVisual_.map(segment), livenessGuard);
        this.scaleView = new Scale(mjVisual_.scale(segment), livenessGuard);
        this.rgbaView = new Rgba(mjVisual_.rgba(segment), livenessGuard);
    }

    public MjVisual(MemorySegment segment, BooleanSupplier livenessGuard) {
        this.segment = segment;
        this.livenessGuard = livenessGuard != null ? livenessGuard : () -> true;
        this.globalView = new Global(mjVisual_.global(segment), this.livenessGuard);
        this.qualityView = new Quality(mjVisual_.quality(segment), this.livenessGuard);
        this.headlightView = new Headlight(mjVisual_.headlight(segment), this.livenessGuard);
        this.mapView = new Map(mjVisual_.map(segment), this.livenessGuard);
        this.scaleView = new Scale(mjVisual_.scale(segment), this.livenessGuard);
        this.rgbaView = new Rgba(mjVisual_.rgba(segment), this.livenessGuard);
    }

    private void checkOpen() {
        if (!livenessGuard.getAsBoolean() || !segment.scope().isAlive()) {
            throw new IllegalStateException("Underlying MjVisual memory has already been closed.");
        }
    }

    public MjVisual copy() {
        checkOpen();
        MjVisual copy = new MjVisual();
        copy.copyFrom(this);
        return copy;
    }

    public void copyFrom(MjVisual other) {
        checkOpen();
        other.checkOpen();
        MemorySegment.copy(other.segment, 0, this.segment, 0, mjVisual_.sizeof());
    }

    public Global global() {
        checkOpen();
        return globalView;
    }

    public void setGlobal(Global other) {
        checkOpen();
        globalView.copyFrom(other);
    }

    public Quality quality() {
        checkOpen();
        return qualityView;
    }

    public Headlight headlight() {
        checkOpen();
        return headlightView;
    }

    public void setHeadlight(Headlight other) {
        checkOpen();
        headlightView.copyFrom(other);
    }

    public Map map() {
        checkOpen();
        return mapView;
    }

    public Scale scale() {
        checkOpen();
        return scaleView;
    }

    public Rgba rgba() {
        checkOpen();
        return rgbaView;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof MjVisual other)) return false;
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
        return "<MjVisual fovy: " + global().fovy() + ", shadowsize: " + quality().shadowsize() + ">";
    }

    public static final class Global {
        private final MemorySegment segment;
        private final BooleanSupplier livenessGuard;

        Global(MemorySegment segment, BooleanSupplier livenessGuard) {
            this.segment = segment;
            this.livenessGuard = livenessGuard;
        }

        public Global copy() {
            Arena arena = Arena.ofAuto();
            MemorySegment copySeg = arena.allocate(mjVisual_.global.layout());
            MemorySegment.copy(segment, 0, copySeg, 0, mjVisual_.global.sizeof());
            return new Global(copySeg, () -> true);
        }

        public void copyFrom(Global other) {
            MemorySegment.copy(other.segment, 0, this.segment, 0, mjVisual_.global.sizeof());
        }

        public float fovy() {
            if (!livenessGuard.getAsBoolean()) throw new IllegalStateException("Closed");
            return mjVisual_.global.fovy(segment);
        }

        public void setFovy(float fovy) {
            if (!livenessGuard.getAsBoolean()) throw new IllegalStateException("Closed");
            mjVisual_.global.fovy(segment, fovy);
        }
    }

    public static final class Quality {
        private final MemorySegment segment;
        private final BooleanSupplier livenessGuard;

        Quality(MemorySegment segment, BooleanSupplier livenessGuard) {
            this.segment = segment;
            this.livenessGuard = livenessGuard;
        }

        public int shadowsize() {
            if (!livenessGuard.getAsBoolean()) throw new IllegalStateException("Closed");
            return mjVisual_.quality.shadowsize(segment);
        }

        public void setShadowsize(int shadowsize) {
            if (!livenessGuard.getAsBoolean()) throw new IllegalStateException("Closed");
            mjVisual_.quality.shadowsize(segment, shadowsize);
        }
    }

    public static final class Headlight {
        private final MemorySegment segment;
        private final BooleanSupplier livenessGuard;
        private final MjFloatVector ambientView;
        private final MjFloatVector diffuseView;
        private final MjFloatVector specularView;

        Headlight(MemorySegment segment, BooleanSupplier livenessGuard) {
            this.segment = segment;
            this.livenessGuard = livenessGuard;
            this.ambientView = new MjFloatVector(mjVisual_.headlight.ambient(segment), 3, livenessGuard);
            this.diffuseView = new MjFloatVector(mjVisual_.headlight.diffuse(segment), 3, livenessGuard);
            this.specularView = new MjFloatVector(mjVisual_.headlight.specular(segment), 3, livenessGuard);
        }

        public Headlight copy() {
            Arena arena = Arena.ofAuto();
            MemorySegment copySeg = arena.allocate(mjVisual_.headlight.layout());
            MemorySegment.copy(segment, 0, copySeg, 0, mjVisual_.headlight.sizeof());
            return new Headlight(copySeg, () -> true);
        }

        public void copyFrom(Headlight other) {
            MemorySegment.copy(other.segment, 0, this.segment, 0, mjVisual_.headlight.sizeof());
        }

        public MjFloatVector ambient() {
            return ambientView;
        }

        public void setAmbient(float r, float g, float b) {
            ambientView.copyFrom(r, g, b);
        }

        public MjFloatVector diffuse() {
            return diffuseView;
        }

        public void setDiffuse(float r, float g, float b) {
            diffuseView.copyFrom(r, g, b);
        }

        public MjFloatVector specular() {
            return specularView;
        }
    }

    public static final class Map {
        private final MemorySegment segment;
        private final BooleanSupplier livenessGuard;

        Map(MemorySegment segment, BooleanSupplier livenessGuard) {
            this.segment = segment;
            this.livenessGuard = livenessGuard;
        }

        public float stiffness() {
            if (!livenessGuard.getAsBoolean()) throw new IllegalStateException("Closed");
            return mjVisual_.map.stiffness(segment);
        }

        public void setStiffness(float stiffness) {
            if (!livenessGuard.getAsBoolean()) throw new IllegalStateException("Closed");
            mjVisual_.map.stiffness(segment, stiffness);
        }
    }

    public static final class Scale {
        private final MemorySegment segment;
        private final BooleanSupplier livenessGuard;

        Scale(MemorySegment segment, BooleanSupplier livenessGuard) {
            this.segment = segment;
            this.livenessGuard = livenessGuard;
        }

        public float framelength() {
            if (!livenessGuard.getAsBoolean()) throw new IllegalStateException("Closed");
            return mjVisual_.scale.framelength(segment);
        }

        public void setFramelength(float framelength) {
            if (!livenessGuard.getAsBoolean()) throw new IllegalStateException("Closed");
            mjVisual_.scale.framelength(segment, framelength);
        }
    }

    public static final class Rgba {
        private final MemorySegment segment;
        private final BooleanSupplier livenessGuard;
        private final MjFloatVector fogView;

        Rgba(MemorySegment segment, BooleanSupplier livenessGuard) {
            this.segment = segment;
            this.livenessGuard = livenessGuard;
            this.fogView = new MjFloatVector(mjVisual_.rgba.fog(segment), 4, livenessGuard);
        }

        public MjFloatVector fog() {
            return fogView;
        }

        public void setFog(float r, float g, float b, float a) {
            fogView.copyFrom(r, g, b, a);
        }
    }
}
