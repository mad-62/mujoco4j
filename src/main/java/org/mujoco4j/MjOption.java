package org.mujoco4j;

import org.mujoco4j.internal.panama.mjOption_;
import org.mujoco4j.internal.panama.mujoco_h;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.function.BooleanSupplier;

/**
 * Wrapper around {@code mjOption} (physics options), either as a live view into {@link MjModel}
 * or as a standalone value object initialized with {@code mj_defaultOption}.
 */
public final class MjOption {

    private final MemorySegment segment;
    private final BooleanSupplier livenessGuard;
    private final MjDoubleVector gravityView;
    private final MjDoubleVector windView;
    private final MjDoubleVector magneticView;

    /**
     * Creates a standalone {@code MjOption} initialized via {@code mj_defaultOption}.
     */
    public MjOption() {
        MjNativeLoader.ensureLoaded();
        Arena arena = Arena.ofAuto();
        this.segment = arena.allocate(mjOption_.layout());
        this.livenessGuard = () -> true;
        mujoco_h.mj_defaultOption(this.segment);
        this.gravityView = new MjDoubleVector(mjOption_.gravity(segment), 3, livenessGuard);
        this.windView = new MjDoubleVector(mjOption_.wind(segment), 3, livenessGuard);
        this.magneticView = new MjDoubleVector(mjOption_.magnetic(segment), 3, livenessGuard);
    }

    public MjOption(MemorySegment segment, BooleanSupplier livenessGuard) {
        this.segment = segment;
        this.livenessGuard = livenessGuard != null ? livenessGuard : () -> true;
        this.gravityView = new MjDoubleVector(mjOption_.gravity(segment), 3, this.livenessGuard);
        this.windView = new MjDoubleVector(mjOption_.wind(segment), 3, this.livenessGuard);
        this.magneticView = new MjDoubleVector(mjOption_.magnetic(segment), 3, this.livenessGuard);
    }

    private void checkOpen() {
        if (!livenessGuard.getAsBoolean() || !segment.scope().isAlive()) {
            throw new IllegalStateException("Underlying MjOption memory has already been closed.");
        }
    }

    public MjOption copy() {
        checkOpen();
        MjOption copy = new MjOption();
        copy.copyFrom(this);
        return copy;
    }

    public void copyFrom(MjOption other) {
        checkOpen();
        other.checkOpen();
        MemorySegment.copy(other.segment, 0, this.segment, 0, mjOption_.sizeof());
    }

    public double timestep() {
        checkOpen();
        return mjOption_.timestep(segment);
    }

    public void setTimestep(double timestep) {
        checkOpen();
        mjOption_.timestep(segment, timestep);
    }

    public double tolerance() {
        checkOpen();
        return mjOption_.tolerance(segment);
    }

    public void setTolerance(double tolerance) {
        checkOpen();
        mjOption_.tolerance(segment, tolerance);
    }

    public double impratio() {
        checkOpen();
        return mjOption_.impratio(segment);
    }

    public void setImpratio(double impratio) {
        checkOpen();
        mjOption_.impratio(segment, impratio);
    }

    public double density() {
        checkOpen();
        return mjOption_.density(segment);
    }

    public void setDensity(double density) {
        checkOpen();
        mjOption_.density(segment, density);
    }

    public double viscosity() {
        checkOpen();
        return mjOption_.viscosity(segment);
    }

    public void setViscosity(double viscosity) {
        checkOpen();
        mjOption_.viscosity(segment, viscosity);
    }

    public MjDoubleVector gravity() {
        checkOpen();
        return gravityView;
    }

    public void setGravity(double x, double y, double z) {
        checkOpen();
        gravityView.set(0, x);
        gravityView.set(1, y);
        gravityView.set(2, z);
    }

    public void setGravity(double uniformValue) {
        checkOpen();
        gravityView.fill(uniformValue);
    }

    public MjDoubleVector wind() {
        checkOpen();
        return windView;
    }

    public MjDoubleVector magnetic() {
        checkOpen();
        return magneticView;
    }

    public int integrator() {
        checkOpen();
        return mjOption_.integrator(segment);
    }

    public MjIntegrator integratorEnum() {
        return MjIntegrator.fromInt(integrator());
    }

    public void setIntegrator(int integrator) {
        checkOpen();
        mjOption_.integrator(segment, integrator);
    }

    public void setIntegrator(MjIntegrator integrator) {
        setIntegrator(integrator.value());
    }

    public int cone() {
        checkOpen();
        return mjOption_.cone(segment);
    }

    public void setCone(int cone) {
        checkOpen();
        mjOption_.cone(segment, cone);
    }

    public int jacobian() {
        checkOpen();
        return mjOption_.jacobian(segment);
    }

    public void setJacobian(int jacobian) {
        checkOpen();
        mjOption_.jacobian(segment, jacobian);
    }

    public void setJacobian(MjJacobian jacobian) {
        setJacobian(jacobian.value());
    }

    public int solver() {
        checkOpen();
        return mjOption_.solver(segment);
    }

    public void setSolver(int solver) {
        checkOpen();
        mjOption_.solver(segment, solver);
    }

    public int iterations() {
        checkOpen();
        return mjOption_.iterations(segment);
    }

    public void setIterations(int iterations) {
        checkOpen();
        mjOption_.iterations(segment, iterations);
    }

    public int disableflags() {
        checkOpen();
        return mjOption_.disableflags(segment);
    }

    public void setDisableflags(int disableflags) {
        checkOpen();
        mjOption_.disableflags(segment, disableflags);
    }

    public int enableflags() {
        checkOpen();
        return mjOption_.enableflags(segment);
    }

    public void setEnableflags(int enableflags) {
        checkOpen();
        mjOption_.enableflags(segment, enableflags);
    }

    public boolean isEnabled(MjEnableBit bit) {
        return (enableflags() & bit.value()) != 0;
    }

    public boolean isDisabled(MjDisableBit bit) {
        return (disableflags() & bit.value()) != 0;
    }

    public MemorySegment segment() {
        checkOpen();
        return segment;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof MjOption other)) return false;
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
        return "<MjOption timestep: " + timestep() + ", tolerance: " + tolerance() + ", gravity: " + gravity() + ">";
    }
}
