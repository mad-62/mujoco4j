package org.mujoco4j;

import org.mujoco4j.internal.panama.mjData_;
import org.mujoco4j.internal.panama.mjSolverStat_;
import org.mujoco4j.internal.panama.mjTimerStat_;
import org.mujoco4j.internal.panama.mjWarningStat_;
import org.mujoco4j.internal.panama.mujoco_h;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * High-level, safe, zero-copy wrapper around MuJoCo's {@code mjData} struct.
 * <p>
 * Retains a strong reference to its parent {@link MjModel} and enforces that both
 * {@code MjData} and its parent {@code MjModel} remain alive before any native memory access.
 */
public final class MjData implements AutoCloseable {

    static {
        MjNativeLoader.ensureLoaded();
    }

    private final MjModel model;
    private final Arena arena;
    private final MemorySegment segment;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    // Cached static buffer vector views (d->buffer pointers remain constant for life of MjData)
    private final MjDoubleVector qposVec;
    private final MjDoubleVector qvelVec;
    private final MjDoubleVector actVec;
    private final MjDoubleVector historyVec;
    private final MjDoubleVector qaccWarmstartVec;
    private final MjDoubleVector ctrlVec;
    private final MjDoubleVector qfrcAppliedVec;
    private final MjDoubleVector xfrcAppliedVec;
    private final MjByteVector eqActiveVec;
    private final MjDoubleVector mocapPosVec;
    private final MjDoubleVector mocapQuatVec;
    private final MjDoubleVector qaccVec;
    private final MjDoubleVector actDotVec;
    private final MjDoubleVector userdataVec;
    private final MjDoubleVector sensordataVec;
    private final MjDoubleVector xposVec;
    private final MjDoubleVector xquatVec;
    private final MjDoubleVector xmatVec;
    private final MjDoubleVector xiposVec;
    private final MjDoubleVector ximatVec;
    private final MjDoubleVector xanchorVec;
    private final MjDoubleVector xaxisVec;
    private final MjDoubleVector geomXposVec;
    private final MjDoubleVector geomXmatVec;
    private final MjDoubleVector siteXposVec;
    private final MjDoubleVector siteXmatVec;
    private final MjDoubleVector camXposVec;
    private final MjDoubleVector camXmatVec;
    private final MjDoubleVector subtreeComVec;
    private final MjDoubleVector cdofVec;
    private final MjDoubleVector cinertVec;
    private final MjDoubleVector tenLengthVec;
    private final MjDoubleVector actuatorLengthVec;
    private final MjDoubleVector qMVec;
    private final MjDoubleVector qLDVec;
    private final MjDoubleVector qLDiagInvVec;
    private final MjDoubleVector tenVelocityVec;
    private final MjDoubleVector actuatorVelocityVec;
    private final MjDoubleVector cvelVec;
    private final MjDoubleVector cdofDotVec;
    private final MjDoubleVector qfrcBiasVec;
    private final MjDoubleVector qfrcSpringVec;
    private final MjDoubleVector qfrcDamperVec;
    private final MjDoubleVector qfrcGravcompVec;
    private final MjDoubleVector qfrcFluidVec;
    private final MjDoubleVector qfrcPassiveVec;
    private final MjDoubleVector subtreeLinvelVec;
    private final MjDoubleVector subtreeAngmomVec;
    private final MjDoubleVector actuatorForceVec;
    private final MjDoubleVector qfrcActuatorVec;
    private final MjDoubleVector qfrcSmoothVec;
    private final MjDoubleVector qaccSmoothVec;
    private final MjDoubleVector qfrcConstraintVec;
    private final MjDoubleVector qfrcInverseVec;
    private final MjDoubleVector caccVec;
    private final MjDoubleVector cfrcIntVec;
    private final MjDoubleVector cfrcExtVec;

    MjData(MjModel model) {
        this(model, mujoco_h.mj_makeData(model.segment()));
    }

    private MjData(MjModel model, MemorySegment rawDataPtr) {
        Objects.requireNonNull(model, "model must not be null");
        model.checkOpen();
        if (rawDataPtr == null || rawDataPtr.equals(MemorySegment.NULL)) {
            throw new MjException("Failed to allocate mjData (null pointer returned)");
        }
        this.model = model;
        this.arena = Arena.ofShared();
        this.segment = rawDataPtr.reinterpret(mjData_.sizeof(), arena, mujoco_h::mj_deleteData);
        MjCallbacks.registerData(this);

        int nq = model.nq();
        int nv = model.nv();
        int nu = model.nu();
        int nout = model.nout();
        int na = model.na();
        int nbody = model.nbody();
        int njnt = model.njnt();
        int ngeom = model.ngeom();
        int nsite = model.nsite();
        int ncam = model.ncam();
        int ntendon = model.ntendon();
        int neq = model.neq();
        int nmocap = model.nmocap();
        int nM = model.nM();
        int nC = model.nC();
        int nuserdata = model.nuserdata();
        int nsensordata = model.nsensordata();
        int nhistory = model.nhistory();

        this.qposVec = doubleVec(mjData_.qpos(segment), nq);
        this.qvelVec = doubleVec(mjData_.qvel(segment), nv);
        this.actVec = doubleVec(mjData_.act(segment), na);
        this.historyVec = doubleVec(mjData_.history(segment), nhistory);
        this.qaccWarmstartVec = doubleVec(mjData_.qacc_warmstart(segment), nv);
        this.ctrlVec = doubleVec(mjData_.ctrl(segment), nu);
        this.qfrcAppliedVec = doubleVec(mjData_.qfrc_applied(segment), nv);
        this.xfrcAppliedVec = doubleVec(mjData_.xfrc_applied(segment), 6L * nbody);
        this.eqActiveVec = byteVec(mjData_.eq_active(segment), neq);
        this.mocapPosVec = doubleVec(mjData_.mocap_pos(segment), 3L * nmocap);
        this.mocapQuatVec = doubleVec(mjData_.mocap_quat(segment), 4L * nmocap);
        this.qaccVec = doubleVec(mjData_.qacc(segment), nv);
        this.actDotVec = doubleVec(mjData_.act_dot(segment), na);
        this.userdataVec = doubleVec(mjData_.userdata(segment), nuserdata);
        this.sensordataVec = doubleVec(mjData_.sensordata(segment), nsensordata);
        this.xposVec = doubleVec(mjData_.xpos(segment), 3L * nbody);
        this.xquatVec = doubleVec(mjData_.xquat(segment), 4L * nbody);
        this.xmatVec = doubleVec(mjData_.xmat(segment), 9L * nbody);
        this.xiposVec = doubleVec(mjData_.xipos(segment), 3L * nbody);
        this.ximatVec = doubleVec(mjData_.ximat(segment), 9L * nbody);
        this.xanchorVec = doubleVec(mjData_.xanchor(segment), 3L * njnt);
        this.xaxisVec = doubleVec(mjData_.xaxis(segment), 3L * njnt);
        this.geomXposVec = doubleVec(mjData_.geom_xpos(segment), 3L * ngeom);
        this.geomXmatVec = doubleVec(mjData_.geom_xmat(segment), 9L * ngeom);
        this.siteXposVec = doubleVec(mjData_.site_xpos(segment), 3L * nsite);
        this.siteXmatVec = doubleVec(mjData_.site_xmat(segment), 9L * nsite);
        this.camXposVec = doubleVec(mjData_.cam_xpos(segment), 3L * ncam);
        this.camXmatVec = doubleVec(mjData_.cam_xmat(segment), 9L * ncam);
        this.subtreeComVec = doubleVec(mjData_.subtree_com(segment), 3L * nbody);
        this.cdofVec = doubleVec(mjData_.cdof(segment), 6L * nv);
        this.cinertVec = doubleVec(mjData_.cinert(segment), 10L * nbody);
        this.tenLengthVec = doubleVec(mjData_.ten_length(segment), ntendon);
        this.actuatorLengthVec = doubleVec(mjData_.actuator_length(segment), nout);
        this.qMVec = doubleVec(mjData_.M(segment), nM);
        this.qLDVec = doubleVec(mjData_.qLD(segment), nC);
        this.qLDiagInvVec = doubleVec(mjData_.qLDiagInv(segment), nv);
        this.tenVelocityVec = doubleVec(mjData_.ten_velocity(segment), ntendon);
        this.actuatorVelocityVec = doubleVec(mjData_.actuator_velocity(segment), nout);
        this.cvelVec = doubleVec(mjData_.cvel(segment), 6L * nbody);
        this.cdofDotVec = doubleVec(mjData_.cdof_dot(segment), 6L * nv);
        this.qfrcBiasVec = doubleVec(mjData_.qfrc_bias(segment), nv);
        this.qfrcSpringVec = doubleVec(mjData_.qfrc_spring(segment), nv);
        this.qfrcDamperVec = doubleVec(mjData_.qfrc_damper(segment), nv);
        this.qfrcGravcompVec = doubleVec(mjData_.qfrc_gravcomp(segment), nv);
        this.qfrcFluidVec = doubleVec(mjData_.qfrc_fluid(segment), nv);
        this.qfrcPassiveVec = doubleVec(mjData_.qfrc_passive(segment), nv);
        this.subtreeLinvelVec = doubleVec(mjData_.subtree_linvel(segment), 3L * nbody);
        this.subtreeAngmomVec = doubleVec(mjData_.subtree_angmom(segment), 3L * nbody);
        this.actuatorForceVec = doubleVec(mjData_.actuator_force(segment), nout);
        this.qfrcActuatorVec = doubleVec(mjData_.qfrc_actuator(segment), nv);
        this.qfrcSmoothVec = doubleVec(mjData_.qfrc_smooth(segment), nv);
        this.qaccSmoothVec = doubleVec(mjData_.qacc_smooth(segment), nv);
        this.qfrcConstraintVec = doubleVec(mjData_.qfrc_constraint(segment), nv);
        this.qfrcInverseVec = doubleVec(mjData_.qfrc_inverse(segment), nv);
        this.caccVec = doubleVec(mjData_.cacc(segment), 6L * nbody);
        this.cfrcIntVec = doubleVec(mjData_.cfrc_int(segment), 6L * nbody);
        this.cfrcExtVec = doubleVec(mjData_.cfrc_ext(segment), 6L * nbody);
    }

    private MjDoubleVector doubleVec(MemorySegment ptr, long length) {
        return new MjDoubleVector(ptr, (int) length, this::isOpen);
    }

    private MjIntVector intVec(MemorySegment ptr, long length) {
        return new MjIntVector(ptr, (int) length, this::isOpen);
    }

    private MjByteVector byteVec(MemorySegment ptr, long length) {
        return new MjByteVector(ptr, (int) length, this::isOpen);
    }

    // =========================================================================
    // Lifecycle & Copy Operations
    // =========================================================================

    public MjModel model() {
        checkOpen();
        return model;
    }

    /**
     * Creates a deep copy of this {@code MjData} bound to the same {@link MjModel} (`mj_copyData`).
     */
    public MjData copy() {
        checkOpen();
        MemorySegment copiedPtr = mujoco_h.mj_copyData(MemorySegment.NULL, model.segment(), this.segment);
        return new MjData(model, copiedPtr);
    }

    /**
     * Copies all state and computed fields from {@code src} into this {@code MjData} (`mj_copyData`).
     */
    public void copyFrom(MjData src) {
        checkOpen();
        Objects.requireNonNull(src, "src must not be null");
        src.checkOpen();
        MemorySegment res = mujoco_h.mj_copyData(this.segment, model.segment(), src.segment);
        if (res.equals(MemorySegment.NULL)) {
            throw new MjException("mj_copyData failed");
        }
    }

    /**
     * Prints a human-readable summary of this data to the specified file path (`mj_printData`).
     */
    public void printData(Path path) {
        checkOpen();
        Objects.requireNonNull(path, "path must not be null");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment pathSeg = temp.allocateFrom(path.toString());
            mujoco_h.mj_printData(model.segment(), this.segment, pathSeg);
        }
    }

    /**
     * Prints a human-readable summary of this data with custom float formatting (`mj_printFormattedData`).
     */
    public void printFormattedData(Path path, String floatFormat) {
        checkOpen();
        Objects.requireNonNull(path, "path must not be null");
        Objects.requireNonNull(floatFormat, "floatFormat must not be null");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment pathSeg = temp.allocateFrom(path.toString());
            MemorySegment fmtSeg = temp.allocateFrom(floatFormat);
            mujoco_h.mj_printFormattedData(model.segment(), this.segment, pathSeg, fmtSeg);
        }
    }

    public long address() {
        checkOpen();
        return segment.address();
    }

    MemorySegment segment() {
        checkOpen();
        return segment;
    }

    MemorySegment segmentUnchecked() {
        return segment;
    }

    Arena arena() {
        return arena;
    }

    public boolean isOpen() {
        return !closed.get() && arena.scope().isAlive() && model.isOpen();
    }

    void checkOpen() {
        if (closed.get() || !arena.scope().isAlive()) {
            throw new IllegalStateException("MjData has already been closed");
        }
        if (!model.isOpen()) {
            throw new IllegalStateException("Parent MjModel of this MjData has already been closed");
        }
    }

    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            MjCallbacks.unregisterData(this);
            arena.close();
        }
    }

    // =========================================================================
    // Scalar State & Dynamic Counters
    // =========================================================================

    public double time() {
        checkOpen();
        return mjData_.time(segment);
    }

    public void setTime(double time) {
        checkOpen();
        mjData_.time(segment, time);
    }

    /**
     * Potential energy {@code d->energy[0]} (requires {@code MjEnableBit.ENERGY}).
     */
    public double energyPos() {
        return energy().get(0);
    }

    public double energyPotential() {
        return energyPos();
    }

    /**
     * Kinetic energy {@code d->energy[1]} (requires {@code MjEnableBit.ENERGY}).
     */
    public double energyVel() {
        return energy().get(1);
    }

    public double energyKinetic() {
        return energyVel();
    }

    public double totalEnergy() {
        return energyPos() + energyVel();
    }

    /**
     * Returns the 2-element energy vector {@code [potential, kinetic]}.
     */
    public MjDoubleVector energy() {
        checkOpen();
        return new MjDoubleVector(mjData_.energy(segment), 2, this::isOpen);
    }

    public int ncon() { checkOpen(); return mjData_.ncon(segment); }
    public int ne() { checkOpen(); return mjData_.ne(segment); }
    public int nf() { checkOpen(); return mjData_.nf(segment); }
    public int nl() { checkOpen(); return mjData_.nl(segment); }
    public int nefc() { checkOpen(); return mjData_.nefc(segment); }
    public int nJ() { checkOpen(); return mjData_.nJ(segment); }
    public int nisland() { checkOpen(); return mjData_.nisland(segment); }
    public int nidof() { checkOpen(); return mjData_.nidof(segment); }

    public long narena() { checkOpen(); return mjData_.narena(segment); }
    public long nbuffer() { checkOpen(); return mjData_.nbuffer(segment); }
    public long pstack() { checkOpen(); return mjData_.pstack(segment); }
    public void setPstack(long pstack) { checkOpen(); mjData_.pstack(segment, pstack); }
    public long maxuseStack() { checkOpen(); return mjData_.maxuse_stack(segment); }
    public long maxuseArena() { checkOpen(); return mjData_.maxuse_arena(segment); }
    public int maxuseCon() { checkOpen(); return mjData_.maxuse_con(segment); }
    public int maxuseEfc() { checkOpen(); return mjData_.maxuse_efc(segment); }

    // =========================================================================
    // Static Buffer Views (d->buffer)
    // =========================================================================

    public MjDoubleVector qpos() { checkOpen(); return qposVec; }
    public MjDoubleVector qvel() { checkOpen(); return qvelVec; }
    public MjDoubleVector act() { checkOpen(); return actVec; }
    public MjDoubleVector history() { checkOpen(); return historyVec; }
    public MjDoubleVector qaccWarmstart() { checkOpen(); return qaccWarmstartVec; }
    public MjDoubleVector ctrl() { checkOpen(); return ctrlVec; }
    public MjDoubleVector qfrcApplied() { checkOpen(); return qfrcAppliedVec; }
    public MjDoubleVector xfrcApplied() { checkOpen(); return xfrcAppliedVec; }
    public MjByteVector eqActive() { checkOpen(); return eqActiveVec; }
    public MjDoubleVector mocapPos() { checkOpen(); return mocapPosVec; }
    public MjDoubleVector mocapQuat() { checkOpen(); return mocapQuatVec; }
    public MjDoubleVector qacc() { checkOpen(); return qaccVec; }
    public MjDoubleVector actDot() { checkOpen(); return actDotVec; }
    public MjDoubleVector userdata() { checkOpen(); return userdataVec; }
    public MjDoubleVector sensordata() { checkOpen(); return sensordataVec; }
    public MjDoubleVector xpos() { checkOpen(); return xposVec; }
    public MjDoubleVector xquat() { checkOpen(); return xquatVec; }
    public MjDoubleVector xmat() { checkOpen(); return xmatVec; }
    public MjDoubleVector xipos() { checkOpen(); return xiposVec; }
    public MjDoubleVector ximat() { checkOpen(); return ximatVec; }
    public MjDoubleVector xanchor() { checkOpen(); return xanchorVec; }
    public MjDoubleVector xaxis() { checkOpen(); return xaxisVec; }
    public MjDoubleVector geomXpos() { checkOpen(); return geomXposVec; }
    public MjDoubleVector geomXmat() { checkOpen(); return geomXmatVec; }
    public MjDoubleVector siteXpos() { checkOpen(); return siteXposVec; }
    public MjDoubleVector siteXmat() { checkOpen(); return siteXmatVec; }
    public MjDoubleVector camXpos() { checkOpen(); return camXposVec; }
    public MjDoubleVector camXmat() { checkOpen(); return camXmatVec; }
    public MjDoubleVector subtreeCom() { checkOpen(); return subtreeComVec; }
    public MjDoubleVector cdof() { checkOpen(); return cdofVec; }
    public MjDoubleVector cinert() { checkOpen(); return cinertVec; }
    public MjDoubleVector tenLength() { checkOpen(); return tenLengthVec; }
    public MjDoubleVector actuatorLength() { checkOpen(); return actuatorLengthVec; }
    public MjDoubleVector qM() { checkOpen(); return qMVec; }
    public MjDoubleVector qLD() { checkOpen(); return qLDVec; }
    public MjDoubleVector qLDiagInv() { checkOpen(); return qLDiagInvVec; }
    public MjDoubleVector tenVelocity() { checkOpen(); return tenVelocityVec; }
    public MjDoubleVector actuatorVelocity() { checkOpen(); return actuatorVelocityVec; }
    public MjDoubleVector cvel() { checkOpen(); return cvelVec; }
    public MjDoubleVector cdofDot() { checkOpen(); return cdofDotVec; }
    public MjDoubleVector qfrcBias() { checkOpen(); return qfrcBiasVec; }
    public MjDoubleVector qfrcSpring() { checkOpen(); return qfrcSpringVec; }
    public MjDoubleVector qfrcDamper() { checkOpen(); return qfrcDamperVec; }
    public MjDoubleVector qfrcGravcomp() { checkOpen(); return qfrcGravcompVec; }
    public MjDoubleVector qfrcFluid() { checkOpen(); return qfrcFluidVec; }
    public MjDoubleVector qfrcPassive() { checkOpen(); return qfrcPassiveVec; }
    public MjDoubleVector subtreeLinvel() { checkOpen(); return subtreeLinvelVec; }
    public MjDoubleVector subtreeAngmom() { checkOpen(); return subtreeAngmomVec; }
    public MjDoubleVector actuatorForce() { checkOpen(); return actuatorForceVec; }
    public MjDoubleVector qfrcActuator() { checkOpen(); return qfrcActuatorVec; }
    public MjDoubleVector qfrcSmooth() { checkOpen(); return qfrcSmoothVec; }
    public MjDoubleVector qaccSmooth() { checkOpen(); return qaccSmoothVec; }
    public MjDoubleVector qfrcConstraint() { checkOpen(); return qfrcConstraintVec; }
    public MjDoubleVector qfrcInverse() { checkOpen(); return qfrcInverseVec; }
    public MjDoubleVector cacc() { checkOpen(); return caccVec; }
    public MjDoubleVector cfrcInt() { checkOpen(); return cfrcIntVec; }
    public MjDoubleVector cfrcExt() { checkOpen(); return cfrcExtVec; }

    // Actuator moment sparse / dense arrays in d->buffer
    public MjIntVector momentRownnz() { checkOpen(); return intVec(mjData_.moment_rownnz(segment), model.nout()); }
    public MjIntVector momentRowadr() { checkOpen(); return intVec(mjData_.moment_rowadr(segment), model.nout()); }
    public MjIntVector momentColind() { checkOpen(); return intVec(mjData_.moment_colind(segment), model.nJmom()); }
    public MjDoubleVector actuatorMoment() { checkOpen(); return doubleVec(mjData_.actuator_moment(segment), model.nJmom()); }
    public MjDoubleVector crb() { checkOpen(); return doubleVec(mjData_.crb(segment), 10L * model.nbody()); }

    // =========================================================================
    // Dynamic Arena Views (d->arena - re-resolved on every call!)
    // =========================================================================

    /**
     * Returns a dynamic list view over the active contacts ({@code d->contact[0..ncon-1]}) in {@code d->arena}.
     */
    public MjContactList contacts() {
        checkOpen();
        return new MjContactList(this);
    }

    /**
     * Convenience accessor for the {@code index}-th active contact.
     */
    public MjContact contact(int index) {
        return contacts().get(index);
    }

    public MjIntVector efcType() {
        checkOpen();
        return intVec(mjData_.efc_type(segment), mjData_.nefc(segment));
    }

    public MjIntVector efcId() {
        checkOpen();
        return intVec(mjData_.efc_id(segment), mjData_.nefc(segment));
    }

    public MjDoubleVector efcJ() {
        checkOpen();
        int len = mujoco_h.mj_isSparse(model.segment()) != 0
                ? mjData_.nJ(segment)
                : mjData_.nefc(segment) * model.nv();
        return doubleVec(mjData_.efc_J(segment), len);
    }

    public MjDoubleVector efcPos() {
        checkOpen();
        return doubleVec(mjData_.efc_pos(segment), mjData_.nefc(segment));
    }

    public MjDoubleVector efcMargin() {
        checkOpen();
        return doubleVec(mjData_.efc_margin(segment), mjData_.nefc(segment));
    }

    public MjDoubleVector efcFrictionloss() {
        checkOpen();
        return doubleVec(mjData_.efc_frictionloss(segment), mjData_.nefc(segment));
    }

    public MjDoubleVector efcD() {
        checkOpen();
        return doubleVec(mjData_.efc_D(segment), mjData_.nefc(segment));
    }

    public MjDoubleVector efcR() {
        checkOpen();
        return doubleVec(mjData_.efc_R(segment), mjData_.nefc(segment));
    }

    public MjDoubleVector efcB() {
        checkOpen();
        return doubleVec(mjData_.efc_b(segment), mjData_.nefc(segment));
    }

    public MjDoubleVector efcForce() {
        checkOpen();
        return doubleVec(mjData_.efc_force(segment), mjData_.nefc(segment));
    }

    public MjIntVector efcState() {
        checkOpen();
        return intVec(mjData_.efc_state(segment), mjData_.nefc(segment));
    }

    public MjDoubleVector efcVel() {
        checkOpen();
        return doubleVec(mjData_.efc_vel(segment), mjData_.nefc(segment));
    }

    public MjDoubleVector efcAref() {
        checkOpen();
        return doubleVec(mjData_.efc_aref(segment), mjData_.nefc(segment));
    }

    public MjIntVector tendonEfcadr() {
        checkOpen();
        return intVec(mjData_.tendon_efcadr(segment), model.ntendon());
    }

    public MjIntVector dofIsland() {
        checkOpen();
        int len = mjData_.nisland(segment) > 0 ? model.nv() : 0;
        return intVec(mjData_.dof_island(segment), len);
    }

    public MjIntVector efcIsland() {
        checkOpen();
        int len = mjData_.nisland(segment) > 0 ? mjData_.nefc(segment) : 0;
        return intVec(mjData_.efc_island(segment), len);
    }

    public MjIntVector treeIsland() {
        checkOpen();
        int len = mjData_.nisland(segment) > 0 ? model.ntree() : 0;
        return intVec(mjData_.tree_island(segment), len);
    }

    // =========================================================================
    // Warnings, Timers & Solver Statistics
    // =========================================================================

    public MjWarningStat warning(MjWarning warning) {
        Objects.requireNonNull(warning, "warning must not be null");
        return warning(warning.code());
    }

    public MjWarningStat warning(int index) {
        checkOpen();
        if (index < 0 || index >= MjConstants.mjNWARNING) {
            throw new IndexOutOfBoundsException("Warning index out of bounds: " + index);
        }
        MemorySegment warnSlice = mjWarningStat_.asSlice(mjData_.warning(segment), index);
        return new MjWarningStat(warnSlice, this::isOpen);
    }

    public MjTimerStat timer(MjTimer timer) {
        Objects.requireNonNull(timer, "timer must not be null");
        return timer(timer.code());
    }

    public MjTimerStat timer(int index) {
        checkOpen();
        if (index < 0 || index >= MjConstants.mjNTIMER) {
            throw new IndexOutOfBoundsException("Timer index out of bounds: " + index);
        }
        MemorySegment timerSlice = mjTimerStat_.asSlice(mjData_.timer(segment), index);
        return new MjTimerStat(timerSlice, this::isOpen);
    }

    public MjSolverStat solver(int index) {
        checkOpen();
        int maxLen = MjConstants.mjNSOLVER * MjConstants.mjNISLAND;
        if (index < 0 || index >= maxLen) {
            throw new IndexOutOfBoundsException("Solver stat index out of bounds: " + index);
        }
        MemorySegment solverSlice = mjSolverStat_.asSlice(mjData_.solver(segment), index);
        return new MjSolverStat(solverSlice, this::isOpen);
    }

    public MjIntVector solverNiter() {
        checkOpen();
        return intVec(mjData_.solver_niter(segment), MjConstants.mjNISLAND);
    }

    public MjIntVector solverNnz() {
        checkOpen();
        return intVec(mjData_.solver_nnz(segment), MjConstants.mjNISLAND);
    }

    public MjDoubleVector solverFwdinv() {
        checkOpen();
        return doubleVec(mjData_.solver_fwdinv(segment), 2);
    }

    // =========================================================================
    // Named & Indexed Element Views (DeepMind bindings_test.py style)
    // =========================================================================

    public BodyDataView body(int id) {
        checkOpen();
        if (id < 0 || id >= model.nbody()) {
            throw new IllegalArgumentException("Invalid body id " + id + " (nbody=" + model.nbody() + ")");
        }
        return new BodyDataView(id);
    }

    public BodyDataView body(String name) {
        checkOpen();
        int id = model.name2id(MjObjType.BODY, name);
        if (id < 0) {
            throw new IllegalArgumentException("Unknown body name: '" + name + "'");
        }
        return new BodyDataView(id);
    }

    public JointDataView joint(int id) {
        checkOpen();
        if (id < 0 || id >= model.njnt()) {
            throw new IllegalArgumentException("Invalid joint id " + id + " (njnt=" + model.njnt() + ")");
        }
        return new JointDataView(id);
    }

    public JointDataView joint(String name) {
        checkOpen();
        int id = model.name2id(MjObjType.JOINT, name);
        if (id < 0) {
            throw new IllegalArgumentException("Unknown joint name: '" + name + "'");
        }
        return new JointDataView(id);
    }

    public GeomDataView geom(int id) {
        checkOpen();
        if (id < 0 || id >= model.ngeom()) {
            throw new IllegalArgumentException("Invalid geom id " + id + " (ngeom=" + model.ngeom() + ")");
        }
        return new GeomDataView(id);
    }

    public GeomDataView geom(String name) {
        checkOpen();
        int id = model.name2id(MjObjType.GEOM, name);
        if (id < 0) {
            throw new IllegalArgumentException("Unknown geom name: '" + name + "'");
        }
        return new GeomDataView(id);
    }

    public SiteDataView site(int id) {
        checkOpen();
        if (id < 0 || id >= model.nsite()) {
            throw new IllegalArgumentException("Invalid site id " + id + " (nsite=" + model.nsite() + ")");
        }
        return new SiteDataView(id);
    }

    public SiteDataView site(String name) {
        checkOpen();
        int id = model.name2id(MjObjType.SITE, name);
        if (id < 0) {
            throw new IllegalArgumentException("Unknown site name: '" + name + "'");
        }
        return new SiteDataView(id);
    }

    public ActuatorDataView actuator(int id) {
        checkOpen();
        if (id < 0 || id >= model.nactuator()) {
            throw new IllegalArgumentException("Invalid actuator id " + id + " (nactuator=" + model.nactuator() + ")");
        }
        return new ActuatorDataView(id);
    }

    public ActuatorDataView actuator(String name) {
        checkOpen();
        int id = model.name2id(MjObjType.ACTUATOR, name);
        if (id < 0) {
            throw new IllegalArgumentException("Unknown actuator name: '" + name + "'");
        }
        return new ActuatorDataView(id);
    }

    public SensorDataView sensor(int id) {
        checkOpen();
        if (id < 0 || id >= model.nsensor()) {
            throw new IllegalArgumentException("Invalid sensor id " + id + " (nsensor=" + model.nsensor() + ")");
        }
        return new SensorDataView(id);
    }

    public SensorDataView sensor(String name) {
        checkOpen();
        int id = model.name2id(MjObjType.SENSOR, name);
        if (id < 0) {
            throw new IllegalArgumentException("Unknown sensor name: '" + name + "'");
        }
        return new SensorDataView(id);
    }

    // =========================================================================
    // Data View Inner Classes
    // =========================================================================

    public final class BodyDataView {
        private final int id;
        private BodyDataView(int id) { this.id = id; }
        public int id() { checkOpen(); return id; }
        public String name() { checkOpen(); String n = model.id2name(MjObjType.BODY, id); return n != null ? n : ""; }
        public MjDoubleVector xpos() { return MjData.this.xpos().slice(3 * id, 3); }
        public MjDoubleVector xquat() { return MjData.this.xquat().slice(4 * id, 4); }
        public MjDoubleVector xmat() { return MjData.this.xmat().slice(9 * id, 9); }
        public MjDoubleVector xipos() { return MjData.this.xipos().slice(3 * id, 3); }
        public MjDoubleVector ximat() { return MjData.this.ximat().slice(9 * id, 9); }
        public MjDoubleVector xfrcApplied() { return MjData.this.xfrcApplied().slice(6 * id, 6); }
        public MjDoubleVector cvel() { return MjData.this.cvel().slice(6 * id, 6); }
        public MjDoubleVector cacc() { return MjData.this.cacc().slice(6 * id, 6); }
        public MjDoubleVector cfrcInt() { return MjData.this.cfrcInt().slice(6 * id, 6); }
        public MjDoubleVector cfrcExt() { return MjData.this.cfrcExt().slice(6 * id, 6); }
    }

    public final class JointDataView {
        private final int id;
        private JointDataView(int id) { this.id = id; }
        public int id() { checkOpen(); return id; }
        public String name() { checkOpen(); String n = model.id2name(MjObjType.JOINT, id); return n != null ? n : ""; }
        private int qposDim() {
            int type = model.jntType().get(id);
            if (type == MjConstants.mjJNT_FREE) return 7;
            if (type == MjConstants.mjJNT_BALL) return 4;
            return 1;
        }
        private int dofDim() {
            int type = model.jntType().get(id);
            if (type == MjConstants.mjJNT_FREE) return 6;
            if (type == MjConstants.mjJNT_BALL) return 3;
            return 1;
        }
        public MjDoubleVector qpos() {
            return MjData.this.qpos().slice(model.jntQposadr().get(id), qposDim());
        }
        public MjDoubleVector qvel() {
            return MjData.this.qvel().slice(model.jntDofadr().get(id), dofDim());
        }
        public MjDoubleVector qacc() {
            return MjData.this.qacc().slice(model.jntDofadr().get(id), dofDim());
        }
        public MjDoubleVector qfrcApplied() {
            return MjData.this.qfrcApplied().slice(model.jntDofadr().get(id), dofDim());
        }
        public MjDoubleVector qfrcConstraint() {
            return MjData.this.qfrcConstraint().slice(model.jntDofadr().get(id), dofDim());
        }
        public MjDoubleVector qfrcSpring() {
            return MjData.this.qfrcSpring().slice(model.jntDofadr().get(id), dofDim());
        }
        public MjDoubleVector qfrcDamper() {
            return MjData.this.qfrcDamper().slice(model.jntDofadr().get(id), dofDim());
        }
        public MjDoubleVector xanchor() { return MjData.this.xanchor().slice(3 * id, 3); }
        public MjDoubleVector xaxis() { return MjData.this.xaxis().slice(3 * id, 3); }
    }

    public final class GeomDataView {
        private final int id;
        private GeomDataView(int id) { this.id = id; }
        public int id() { checkOpen(); return id; }
        public String name() { checkOpen(); String n = model.id2name(MjObjType.GEOM, id); return n != null ? n : ""; }
        public MjDoubleVector xpos() { return MjData.this.geomXpos().slice(3 * id, 3); }
        public MjDoubleVector xmat() { return MjData.this.geomXmat().slice(9 * id, 9); }
    }

    public final class SiteDataView {
        private final int id;
        private SiteDataView(int id) { this.id = id; }
        public int id() { checkOpen(); return id; }
        public String name() { checkOpen(); String n = model.id2name(MjObjType.SITE, id); return n != null ? n : ""; }
        public MjDoubleVector xpos() { return MjData.this.siteXpos().slice(3 * id, 3); }
        public MjDoubleVector xmat() { return MjData.this.siteXmat().slice(9 * id, 9); }
    }

    public final class ActuatorDataView {
        private final int id;
        private ActuatorDataView(int id) { this.id = id; }
        public int id() { checkOpen(); return id; }
        public String name() { checkOpen(); String n = model.id2name(MjObjType.ACTUATOR, id); return n != null ? n : ""; }
        public MjDoubleVector ctrl() {
            int adr = model.actuatorCtrladr().get(id);
            int num = model.actuatorCtrlnum().get(id);
            return adr >= 0 && num > 0
                    ? MjData.this.ctrl().slice(adr, num)
                    : MjData.this.ctrl().slice(id, 1);
        }
        public MjDoubleVector length() {
            int adr = model.actuatorOutadr().get(id);
            int num = model.actuatorOutnum().get(id);
            return adr >= 0 && num > 0
                    ? MjData.this.actuatorLength().slice(adr, num)
                    : MjData.this.actuatorLength().slice(id, 1);
        }
        public MjDoubleVector velocity() {
            int adr = model.actuatorOutadr().get(id);
            int num = model.actuatorOutnum().get(id);
            return adr >= 0 && num > 0
                    ? MjData.this.actuatorVelocity().slice(adr, num)
                    : MjData.this.actuatorVelocity().slice(id, 1);
        }
        public MjDoubleVector force() {
            int adr = model.actuatorOutadr().get(id);
            int num = model.actuatorOutnum().get(id);
            return adr >= 0 && num > 0
                    ? MjData.this.actuatorForce().slice(adr, num)
                    : MjData.this.actuatorForce().slice(id, 1);
        }
    }

    public final class SensorDataView {
        private final int id;
        private SensorDataView(int id) { this.id = id; }
        public int id() { checkOpen(); return id; }
        public String name() { checkOpen(); String n = model.id2name(MjObjType.SENSOR, id); return n != null ? n : ""; }
        public MjDoubleVector data() {
            int adr = model.sensorAdr().get(id);
            int dim = model.sensorDim().get(id);
            return MjData.this.sensordata().slice(adr, dim);
        }
    }
}
