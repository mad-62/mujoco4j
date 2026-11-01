package org.mujoco4j;

import org.mujoco4j.internal.panama.mujoco_h;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Objects;

/**
 * High-level physics simulation, dynamics, state, Jacobian, finite-difference,
 * and delay-history functions wrapping MuJoCo's {@code mj_*} and {@code mjd_*} APIs.
 */
public final class MjSim {

    static {
        MjNativeLoader.ensureLoaded();
    }

    private MjSim() {}

    // =========================================================================
    // Version & Header Metadata
    // =========================================================================

    /**
     * Returns the runtime MuJoCo version integer (e.g. {@code 3017000} for 3.17.0) (`mj_version`).
     */
    public static int version() {
        return mujoco_h.mj_version();
    }

    /**
     * Returns the runtime MuJoCo version string (e.g. {@code "3.17.0"}) (`mj_versionString`).
     */
    public static String versionString() {
        MemorySegment ptr = mujoco_h.mj_versionString();
        return ptr.equals(MemorySegment.NULL) ? "" : ptr.reinterpret(Long.MAX_VALUE).getString(0);
    }

    // =========================================================================
    // Main Simulation Pipeline
    // =========================================================================

    /**
     * Advances the simulation by one timestep (`mj_step`).
     */
    public static void step(MjModel m, MjData d) {
        checkArgs(m, d);
        mujoco_h.mj_step(m.segment(), d.segment());
        MjCallbacks.rethrowIfPending();
    }

    /**
     * Advances the simulation by {@code nstep} timesteps in a tight loop.
     */
    public static void step(MjModel m, MjData d, int nstep) {
        checkArgs(m, d);
        if (nstep < 0) {
            throw new IllegalArgumentException("nstep must be >= 0: " + nstep);
        }
        MemorySegment mSeg = m.segment();
        MemorySegment dSeg = d.segment();
        for (int i = 0; i < nstep; i++) {
            mujoco_h.mj_step(mSeg, dSeg);
            MjCallbacks.rethrowIfPending();
        }
    }

    /**
     * First-part of two-stage stepping: runs position/velocity kinematics and control preparation (`mj_step1`).
     */
    public static void step1(MjModel m, MjData d) {
        checkArgs(m, d);
        mujoco_h.mj_step1(m.segment(), d.segment());
        MjCallbacks.rethrowIfPending();
    }

    /**
     * Second-part of two-stage stepping: runs actuation, acceleration, and integration (`mj_step2`).
     */
    public static void step2(MjModel m, MjData d) {
        checkArgs(m, d);
        mujoco_h.mj_step2(m.segment(), d.segment());
        MjCallbacks.rethrowIfPending();
    }

    /**
     * Computes forward dynamics without integrating in time (`mj_forward`).
     */
    public static void forward(MjModel m, MjData d) {
        checkArgs(m, d);
        mujoco_h.mj_forward(m.segment(), d.segment());
        MjCallbacks.rethrowIfPending();
    }

    /**
     * Computes forward dynamics skipping stages up to {@code skipstage} (`mj_forwardSkip`).
     */
    public static void forwardSkip(MjModel m, MjData d, MjStage skipstage, int skipsensor) {
        Objects.requireNonNull(skipstage, "skipstage must not be null");
        forwardSkip(m, d, skipstage.code(), skipsensor);
    }

    /**
     * Computes forward dynamics skipping stages up to {@code skipstage} (`mj_forwardSkip`).
     */
    public static void forwardSkip(MjModel m, MjData d, int skipstage, int skipsensor) {
        checkArgs(m, d);
        mujoco_h.mj_forwardSkip(m.segment(), d.segment(), skipstage, skipsensor);
        MjCallbacks.rethrowIfPending();
    }

    /**
     * Computes inverse dynamics (`mj_inverse`).
     */
    public static void inverse(MjModel m, MjData d) {
        checkArgs(m, d);
        mujoco_h.mj_inverse(m.segment(), d.segment());
    }

    /**
     * Computes inverse dynamics skipping stages up to {@code skipstage} (`mj_inverseSkip`).
     */
    public static void inverseSkip(MjModel m, MjData d, MjStage skipstage, int skipsensor) {
        Objects.requireNonNull(skipstage, "skipstage must not be null");
        inverseSkip(m, d, skipstage.code(), skipsensor);
    }

    /**
     * Computes inverse dynamics skipping stages up to {@code skipstage} (`mj_inverseSkip`).
     */
    public static void inverseSkip(MjModel m, MjData d, int skipstage, int skipsensor) {
        checkArgs(m, d);
        mujoco_h.mj_inverseSkip(m.segment(), d.segment(), skipstage, skipsensor);
    }

    // =========================================================================
    // Reset & Keyframes
    // =========================================================================

    /**
     * Resets {@code MjData} to the model's default configuration {@code qpos0} (`mj_resetData`).
     */
    public static void resetData(MjModel m, MjData d) {
        checkArgs(m, d);
        mujoco_h.mj_resetData(m.segment(), d.segment());
    }

    /**
     * Resets {@code MjData} with debug value filling (`mj_resetDataDebug`).
     */
    public static void resetDataDebug(MjModel m, MjData d, byte debugValue) {
        checkArgs(m, d);
        mujoco_h.mj_resetDataDebug(m.segment(), d.segment(), debugValue);
    }

    /**
     * Resets {@code MjData} to the specified keyframe index (`mj_resetDataKeyframe`).
     */
    public static void resetDataKeyframe(MjModel m, MjData d, int key) {
        checkArgs(m, d);
        if (key < 0 || key >= m.nkey()) {
            throw new IllegalArgumentException("Invalid keyframe index " + key + " (nkey=" + m.nkey() + ")");
        }
        mujoco_h.mj_resetDataKeyframe(m.segment(), d.segment(), key);
    }

    /**
     * Copies the current state of {@code MjData} into keyframe {@code key} of {@code MjModel} (`mj_setKeyframe`).
     */
    public static void setKeyframe(MjModel m, MjData d, int key) {
        checkArgs(m, d);
        if (key < 0 || key >= m.nkey()) {
            throw new IllegalArgumentException("Invalid keyframe index " + key + " (nkey=" + m.nkey() + ")");
        }
        mujoco_h.mj_setKeyframe(m.segment(), d.segment(), key);
    }

    /**
     * Resets all simulation callbacks to {@code NULL} (`mj_resetCallbacks`).
     */
    public static void resetCallbacks() {
        MjCallbacks.clearAll();
    }

    // =========================================================================
    // Sub-pipeline Stages & Integrators
    // =========================================================================

    public static void kinematics(MjModel m, MjData d) { checkArgs(m, d); mujoco_h.mj_kinematics(m.segment(), d.segment()); }
    public static void comPos(MjModel m, MjData d) { checkArgs(m, d); mujoco_h.mj_comPos(m.segment(), d.segment()); }
    public static void camlight(MjModel m, MjData d) { checkArgs(m, d); mujoco_h.mj_camlight(m.segment(), d.segment()); }
    public static void flex(MjModel m, MjData d) { checkArgs(m, d); mujoco_h.mj_flex(m.segment(), d.segment()); }
    public static void tendon(MjModel m, MjData d) { checkArgs(m, d); mujoco_h.mj_tendon(m.segment(), d.segment()); }
    public static void transmission(MjModel m, MjData d) { checkArgs(m, d); mujoco_h.mj_transmission(m.segment(), d.segment()); }
    public static void crb(MjModel m, MjData d) { checkArgs(m, d); mujoco_h.mj_crb(m.segment(), d.segment()); }
    public static void makeM(MjModel m, MjData d) { checkArgs(m, d); mujoco_h.mj_makeM(m.segment(), d.segment()); }
    public static void factorM(MjModel m, MjData d) { checkArgs(m, d); mujoco_h.mj_factorM(m.segment(), d.segment()); }
    public static void collision(MjModel m, MjData d) { checkArgs(m, d); mujoco_h.mj_collision(m.segment(), d.segment()); }
    public static void makeConstraint(MjModel m, MjData d) { checkArgs(m, d); mujoco_h.mj_makeConstraint(m.segment(), d.segment()); }
    public static void island(MjModel m, MjData d) { checkArgs(m, d); mujoco_h.mj_island(m.segment(), d.segment()); }
    public static void projectConstraint(MjModel m, MjData d) { checkArgs(m, d); mujoco_h.mj_projectConstraint(m.segment(), d.segment()); }
    public static void referenceConstraint(MjModel m, MjData d) { checkArgs(m, d); mujoco_h.mj_referenceConstraint(m.segment(), d.segment()); }
    public static void constraintUpdate(MjModel m, MjData d, double[] jar, double[] cost, int flgConeHessian) {
        checkArgs(m, d);
        int nefc = d.nefc();
        if (jar == null || jar.length < nefc) {
            throw new IllegalArgumentException("jar length must be >= nefc (" + nefc + ")");
        }
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment jarSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, jar);
            MemorySegment costSeg = cost != null && cost.length >= 1 ? temp.allocate(ValueLayout.JAVA_DOUBLE) : MemorySegment.NULL;
            mujoco_h.mj_constraintUpdate(m.segment(), d.segment(), jarSeg, costSeg, flgConeHessian);
            if (cost != null && cost.length >= 1) {
                cost[0] = costSeg.get(ValueLayout.JAVA_DOUBLE, 0);
            }
        }
    }

    public static void fwdPosition(MjModel m, MjData d) { checkArgs(m, d); mujoco_h.mj_fwdPosition(m.segment(), d.segment()); }
    public static void fwdVelocity(MjModel m, MjData d) { checkArgs(m, d); mujoco_h.mj_fwdVelocity(m.segment(), d.segment()); }
    public static void fwdActuation(MjModel m, MjData d) { checkArgs(m, d); mujoco_h.mj_fwdActuation(m.segment(), d.segment()); }
    public static void fwdAcceleration(MjModel m, MjData d) { checkArgs(m, d); mujoco_h.mj_fwdAcceleration(m.segment(), d.segment()); }
    public static void fwdConstraint(MjModel m, MjData d) { checkArgs(m, d); mujoco_h.mj_fwdConstraint(m.segment(), d.segment()); }
    public static void Euler(MjModel m, MjData d) { checkArgs(m, d); mujoco_h.mj_Euler(m.segment(), d.segment()); }
    public static void RungeKutta(MjModel m, MjData d, int N) { checkArgs(m, d); mujoco_h.mj_RungeKutta(m.segment(), d.segment(), N); }
    public static void implicit(MjModel m, MjData d) { checkArgs(m, d); mujoco_h.mj_implicit(m.segment(), d.segment()); }
    public static void invPosition(MjModel m, MjData d) { checkArgs(m, d); mujoco_h.mj_invPosition(m.segment(), d.segment()); }
    public static void invVelocity(MjModel m, MjData d) { checkArgs(m, d); mujoco_h.mj_invVelocity(m.segment(), d.segment()); }
    public static void invConstraint(MjModel m, MjData d) { checkArgs(m, d); mujoco_h.mj_invConstraint(m.segment(), d.segment()); }
    public static void compareFwdInv(MjModel m, MjData d) { checkArgs(m, d); mujoco_h.mj_compareFwdInv(m.segment(), d.segment()); }

    public static void sensorPos(MjModel m, MjData d) { checkArgs(m, d); mujoco_h.mj_sensorPos(m.segment(), d.segment()); }
    public static void sensorVel(MjModel m, MjData d) { checkArgs(m, d); mujoco_h.mj_sensorVel(m.segment(), d.segment()); }
    public static void sensorAcc(MjModel m, MjData d) { checkArgs(m, d); mujoco_h.mj_sensorAcc(m.segment(), d.segment()); }
    public static void energyPos(MjModel m, MjData d) { checkArgs(m, d); mujoco_h.mj_energyPos(m.segment(), d.segment()); }
    public static void energyVel(MjModel m, MjData d) { checkArgs(m, d); mujoco_h.mj_energyVel(m.segment(), d.segment()); }
    public static void checkPos(MjModel m, MjData d) { checkArgs(m, d); mujoco_h.mj_checkPos(m.segment(), d.segment()); }
    public static void checkVel(MjModel m, MjData d) { checkArgs(m, d); mujoco_h.mj_checkVel(m.segment(), d.segment()); }
    public static void checkAcc(MjModel m, MjData d) { checkArgs(m, d); mujoco_h.mj_checkAcc(m.segment(), d.segment()); }
    public static void rne(MjModel m, MjData d, int flgAcc, double[] result) {
        checkArgs(m, d);
        int nv = m.nv();
        if (result == null || result.length < nv) {
            throw new IllegalArgumentException("result length must be >= nv (" + nv + ")");
        }
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment resSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, nv);
            mujoco_h.mj_rne(m.segment(), d.segment(), flgAcc, resSeg);
            MemorySegment.copy(resSeg, ValueLayout.JAVA_DOUBLE, 0, result, 0, nv);
        }
    }
    public static void rnePostConstraint(MjModel m, MjData d) { checkArgs(m, d); mujoco_h.mj_rnePostConstraint(m.segment(), d.segment()); }
    public static void comVel(MjModel m, MjData d) { checkArgs(m, d); mujoco_h.mj_comVel(m.segment(), d.segment()); }
    public static void passive(MjModel m, MjData d) { checkArgs(m, d); mujoco_h.mj_passive(m.segment(), d.segment()); }
    public static void subtreeVel(MjModel m, MjData d) { checkArgs(m, d); mujoco_h.mj_subtreeVel(m.segment(), d.segment()); }

    // =========================================================================
    // State Serialization (mj_stateSize / mj_getState / mj_setState / mj_extractState)
    // =========================================================================

    /**
     * Returns the number of {@code double} elements needed to store the state specified by {@code sig} (`mj_stateSize`).
     */
    public static int stateSize(MjModel m, MjState sig) {
        Objects.requireNonNull(sig, "sig must not be null");
        return stateSize(m, sig.mask());
    }

    /**
     * Returns the number of {@code double} elements needed to store the state specified by {@code sigMask} (`mj_stateSize`).
     */
    public static int stateSize(MjModel m, int sigMask) {
        Objects.requireNonNull(m, "m must not be null");
        m.checkOpen();
        return (int) mujoco_h.mj_stateSize(m.segment(), sigMask);
    }

    /**
     * Copies the simulation state specified by {@code sig} into a newly allocated {@code double[]} (`mj_getState`).
     */
    public static double[] getState(MjModel m, MjData d, MjState sig) {
        Objects.requireNonNull(sig, "sig must not be null");
        return getState(m, d, sig.mask());
    }

    /**
     * Copies the simulation state specified by {@code sigMask} into a newly allocated {@code double[]} (`mj_getState`).
     */
    public static double[] getState(MjModel m, MjData d, int sigMask) {
        int size = stateSize(m, sigMask);
        double[] state = new double[size];
        getState(m, d, state, sigMask);
        return state;
    }

    /**
     * Copies the simulation state specified by {@code sigMask} into {@code state} (`mj_getState`).
     */
    public static void getState(MjModel m, MjData d, double[] state, int sigMask) {
        checkArgs(m, d);
        int size = stateSize(m, sigMask);
        if (state == null || state.length < size) {
            throw new IllegalArgumentException("state array length (" + (state == null ? 0 : state.length) + ") < stateSize (" + size + ")");
        }
        if (size == 0) {
            return;
        }
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment stateSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, size);
            mujoco_h.mj_getState(m.segment(), d.segment(), stateSeg, sigMask);
            MemorySegment.copy(stateSeg, ValueLayout.JAVA_DOUBLE, 0, state, 0, size);
        }
    }

    /**
     * Restores the simulation state specified by {@code sig} from {@code state} (`mj_setState`).
     */
    public static void setState(MjModel m, MjData d, double[] state, MjState sig) {
        Objects.requireNonNull(sig, "sig must not be null");
        setState(m, d, state, sig.mask());
    }

    /**
     * Restores the simulation state specified by {@code sigMask} from {@code state} (`mj_setState`).
     */
    public static void setState(MjModel m, MjData d, double[] state, int sigMask) {
        checkArgs(m, d);
        int size = stateSize(m, sigMask);
        if (state == null || state.length < size) {
            throw new IllegalArgumentException("state array length (" + (state == null ? 0 : state.length) + ") < stateSize (" + size + ")");
        }
        if (size == 0) {
            return;
        }
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment stateSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, state);
            mujoco_h.mj_setState(m.segment(), d.segment(), stateSeg, sigMask);
        }
    }

    /**
     * Extracts a subset of state components {@code dstSig} from a state vector with signature {@code srcSig} (`mj_extractState`).
     */
    public static void extractState(MjModel m, double[] src, int srcSig, double[] dst, int dstSig) {
        Objects.requireNonNull(m, "m must not be null");
        m.checkOpen();
        int srcSize = stateSize(m, srcSig);
        int dstSize = stateSize(m, dstSig);
        if (src == null || src.length < srcSize) {
            throw new IllegalArgumentException("src array length < srcSize (" + srcSize + ")");
        }
        if (dst == null || dst.length < dstSize) {
            throw new IllegalArgumentException("dst array length < dstSize (" + dstSize + ")");
        }
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment srcSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, src);
            MemorySegment dstSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, Math.max(1, dstSize));
            mujoco_h.mj_extractState(m.segment(), srcSeg, srcSig, dstSeg, dstSig);
            if (dstSize > 0) {
                MemorySegment.copy(dstSeg, ValueLayout.JAVA_DOUBLE, 0, dst, 0, dstSize);
            }
        }
    }

    // =========================================================================
    // Mass Matrix Operations (mj_fullM, mj_mulM, mj_mulM2, mj_solveM, mj_solveM2, mj_addM)
    // =========================================================================

    public static boolean isSparse(MjModel m) {
        Objects.requireNonNull(m, "m must not be null");
        m.checkOpen();
        return mujoco_h.mj_isSparse(m.segment()) != 0;
    }

    public static boolean isPyramidal(MjModel m) {
        Objects.requireNonNull(m, "m must not be null");
        m.checkOpen();
        return mujoco_h.mj_isPyramidal(m.segment()) != 0;
    }

    public static boolean isDual(MjModel m) {
        Objects.requireNonNull(m, "m must not be null");
        m.checkOpen();
        return mujoco_h.mj_isDual(m.segment()) != 0;
    }

    /**
     * Converts sparse inertia matrix {@code M} (such as {@code d.qM()}) into a dense {@code nv x nv} matrix {@code dst} (`mj_fullM`).
     */
    public static void fullM(MjModel m, double[] dst, MjDoubleVector M) {
        Objects.requireNonNull(m, "m must not be null");
        Objects.requireNonNull(M, "M must not be null");
        m.checkOpen();
        int nv = m.nv();
        if (dst == null || dst.length < nv * nv) {
            throw new IllegalArgumentException("dst length must be >= nv*nv (" + (nv * nv) + ")");
        }
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment dstSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, (long) nv * nv);
            mujoco_h.mj_fullM(m.segment(), dstSeg, M.segment());
            MemorySegment.copy(dstSeg, ValueLayout.JAVA_DOUBLE, 0, dst, 0, nv * nv);
        }
    }

    /**
     * Converts sparse inertia matrix {@code M} into a dense {@code nv x nv} matrix {@code dst} (`mj_fullM`).
     */
    public static void fullM(MjModel m, double[] dst, double[] M) {
        Objects.requireNonNull(m, "m must not be null");
        m.checkOpen();
        int nv = m.nv();
        int nM = m.nM();
        if (dst == null || dst.length < nv * nv) {
            throw new IllegalArgumentException("dst length must be >= nv*nv (" + (nv * nv) + ")");
        }
        if (M == null || M.length < nM) {
            throw new IllegalArgumentException("M length must be >= nM (" + nM + ")");
        }
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment dstSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, (long) nv * nv);
            MemorySegment mSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, M);
            mujoco_h.mj_fullM(m.segment(), dstSeg, mSeg);
            MemorySegment.copy(dstSeg, ValueLayout.JAVA_DOUBLE, 0, dst, 0, nv * nv);
        }
    }

    /**
     * Converts sparse inertia matrix {@code d->M} into a newly allocated dense {@code nv x nv} array (`mj_fullM`).
     */
    public static double[] fullM(MjModel m, MjData d) {
        checkArgs(m, d);
        int nv = m.nv();
        double[] dst = new double[nv * nv];
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment dstSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, (long) nv * nv);
            mujoco_h.mj_fullM(m.segment(), d.segment(), dstSeg);
            MemorySegment.copy(dstSeg, ValueLayout.JAVA_DOUBLE, 0, dst, 0, nv * nv);
        }
        return dst;
    }

    /**
     * Multiplies a vector by the inertia matrix: {@code res = M * vec} (`mj_mulM`).
     */
    public static void mulM(MjModel m, MjData d, double[] res, double[] vec) {
        checkArgs(m, d);
        int nv = m.nv();
        if (res == null || res.length < nv || vec == null || vec.length < nv) {
            throw new IllegalArgumentException("res and vec must have length >= nv (" + nv + ")");
        }
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment resSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, nv);
            MemorySegment vecSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vec);
            mujoco_h.mj_mulM(m.segment(), d.segment(), resSeg, vecSeg);
            MemorySegment.copy(resSeg, ValueLayout.JAVA_DOUBLE, 0, res, 0, nv);
        }
    }

    /**
     * Multiplies a vector by the square root of the inertia matrix: {@code res = sqrt(M) * vec} (`mj_mulM2`).
     */
    public static void mulM2(MjModel m, MjData d, double[] res, double[] vec) {
        checkArgs(m, d);
        int nv = m.nv();
        if (res == null || res.length < nv || vec == null || vec.length < nv) {
            throw new IllegalArgumentException("res and vec must have length >= nv (" + nv + ")");
        }
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment resSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, nv);
            MemorySegment vecSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vec);
            mujoco_h.mj_mulM2(m.segment(), d.segment(), resSeg, vecSeg);
            MemorySegment.copy(resSeg, ValueLayout.JAVA_DOUBLE, 0, res, 0, nv);
        }
    }

    /**
     * Solves linear system {@code M * x = y} for {@code n} right-hand-side vectors of length {@code nv} (`mj_solveM`).
     */
    public static void solveM(MjModel m, MjData d, double[] x, double[] y, int n) {
        checkArgs(m, d);
        int nv = m.nv();
        int total = n * nv;
        if (x == null || x.length < total || y == null || y.length < total) {
            throw new IllegalArgumentException("x and y must have length >= n*nv (" + total + ")");
        }
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment xSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, total);
            MemorySegment ySeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, y);
            mujoco_h.mj_solveM(m.segment(), d.segment(), xSeg, ySeg, n);
            MemorySegment.copy(xSeg, ValueLayout.JAVA_DOUBLE, 0, x, 0, total);
        }
    }

    /**
     * Solves half of linear system {@code sqrt(M) * x = y} for {@code n} vectors of length {@code nv} (`mj_solveM2`).
     */
    public static void solveM2(MjModel m, MjData d, double[] x, double[] y, double[] sqrtInvD, int n) {
        checkArgs(m, d);
        int nv = m.nv();
        int total = n * nv;
        if (x == null || x.length < total || y == null || y.length < total) {
            throw new IllegalArgumentException("x and y must have length >= n*nv (" + total + ")");
        }
        if (sqrtInvD == null || sqrtInvD.length < nv) {
            throw new IllegalArgumentException("sqrtInvD must have length >= nv (" + nv + ")");
        }
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment xSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, total);
            MemorySegment ySeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, y);
            MemorySegment dSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, sqrtInvD);
            mujoco_h.mj_solveM2(m.segment(), d.segment(), xSeg, ySeg, dSeg, n);
            MemorySegment.copy(xSeg, ValueLayout.JAVA_DOUBLE, 0, x, 0, total);
        }
    }

    /**
     * Adds inertia matrix {@code M} into destination sparse or dense matrix {@code dst} (`mj_addM`).
     */
    public static void addM(MjModel m, MjData d, double[] dst, int[] rownnz, int[] rowadr, int[] colind) {
        checkArgs(m, d);
        Objects.requireNonNull(dst, "dst must not be null");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment dstSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, dst);
            MemorySegment nnzSeg = rownnz != null ? temp.allocateFrom(ValueLayout.JAVA_INT, rownnz) : MemorySegment.NULL;
            MemorySegment adrSeg = rowadr != null ? temp.allocateFrom(ValueLayout.JAVA_INT, rowadr) : MemorySegment.NULL;
            MemorySegment colSeg = colind != null ? temp.allocateFrom(ValueLayout.JAVA_INT, colind) : MemorySegment.NULL;
            mujoco_h.mj_addM(m.segment(), d.segment(), dstSeg, nnzSeg, adrSeg, colSeg);
            MemorySegment.copy(dstSeg, ValueLayout.JAVA_DOUBLE, 0, dst, 0, dst.length);
            if (rownnz != null) MemorySegment.copy(nnzSeg, ValueLayout.JAVA_INT, 0, rownnz, 0, rownnz.length);
            if (rowadr != null) MemorySegment.copy(adrSeg, ValueLayout.JAVA_INT, 0, rowadr, 0, rowadr.length);
            if (colind != null) MemorySegment.copy(colSeg, ValueLayout.JAVA_INT, 0, colind, 0, colind.length);
        }
    }

    // =========================================================================
    // Jacobians & Coordinate Transformations (mj_jac*, mj_mulJac*, mj_local2Global)
    // =========================================================================

    /**
     * Computes the 3xNV translational (`jacp`) and 3xNV rotational (`jacr`) Jacobians at {@code point} on {@code body} (`mj_jac`).
     * Either {@code jacp} or {@code jacr} may be {@code null}.
     */
    public static void jac(MjModel m, MjData d, double[] jacp, double[] jacr, double[] point, int body) {
        checkArgs(m, d);
        if (point == null || point.length < 3) {
            throw new IllegalArgumentException("point must have length >= 3");
        }
        int nv = m.nv();
        checkJacBuffers(jacp, jacr, nv);
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment jpSeg = jacp != null ? temp.allocate(ValueLayout.JAVA_DOUBLE, 3L * nv) : MemorySegment.NULL;
            MemorySegment jrSeg = jacr != null ? temp.allocate(ValueLayout.JAVA_DOUBLE, 3L * nv) : MemorySegment.NULL;
            MemorySegment ptSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, point);
            mujoco_h.mj_jac(m.segment(), d.segment(), jpSeg, jrSeg, ptSeg, body);
            copyOutJac(jpSeg, jacp, jrSeg, jacr, 3 * nv);
        }
    }

    public static void jacBody(MjModel m, MjData d, double[] jacp, double[] jacr, int body) {
        checkArgs(m, d);
        int nv = m.nv();
        checkJacBuffers(jacp, jacr, nv);
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment jpSeg = jacp != null ? temp.allocate(ValueLayout.JAVA_DOUBLE, 3L * nv) : MemorySegment.NULL;
            MemorySegment jrSeg = jacr != null ? temp.allocate(ValueLayout.JAVA_DOUBLE, 3L * nv) : MemorySegment.NULL;
            mujoco_h.mj_jacBody(m.segment(), d.segment(), jpSeg, jrSeg, body);
            copyOutJac(jpSeg, jacp, jrSeg, jacr, 3 * nv);
        }
    }

    public static void jacBodyCom(MjModel m, MjData d, double[] jacp, double[] jacr, int body) {
        checkArgs(m, d);
        int nv = m.nv();
        checkJacBuffers(jacp, jacr, nv);
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment jpSeg = jacp != null ? temp.allocate(ValueLayout.JAVA_DOUBLE, 3L * nv) : MemorySegment.NULL;
            MemorySegment jrSeg = jacr != null ? temp.allocate(ValueLayout.JAVA_DOUBLE, 3L * nv) : MemorySegment.NULL;
            mujoco_h.mj_jacBodyCom(m.segment(), d.segment(), jpSeg, jrSeg, body);
            copyOutJac(jpSeg, jacp, jrSeg, jacr, 3 * nv);
        }
    }

    public static void jacSubtreeCom(MjModel m, MjData d, double[] jacp, int body) {
        checkArgs(m, d);
        int nv = m.nv();
        if (jacp == null || jacp.length < 3 * nv) {
            throw new IllegalArgumentException("jacp length must be >= 3*nv (" + (3 * nv) + ")");
        }
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment jpSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, 3L * nv);
            mujoco_h.mj_jacSubtreeCom(m.segment(), d.segment(), jpSeg, body);
            MemorySegment.copy(jpSeg, ValueLayout.JAVA_DOUBLE, 0, jacp, 0, 3 * nv);
        }
    }

    public static void jacGeom(MjModel m, MjData d, double[] jacp, double[] jacr, int geom) {
        checkArgs(m, d);
        int nv = m.nv();
        checkJacBuffers(jacp, jacr, nv);
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment jpSeg = jacp != null ? temp.allocate(ValueLayout.JAVA_DOUBLE, 3L * nv) : MemorySegment.NULL;
            MemorySegment jrSeg = jacr != null ? temp.allocate(ValueLayout.JAVA_DOUBLE, 3L * nv) : MemorySegment.NULL;
            mujoco_h.mj_jacGeom(m.segment(), d.segment(), jpSeg, jrSeg, geom);
            copyOutJac(jpSeg, jacp, jrSeg, jacr, 3 * nv);
        }
    }

    public static void jacSite(MjModel m, MjData d, double[] jacp, double[] jacr, int site) {
        checkArgs(m, d);
        int nv = m.nv();
        checkJacBuffers(jacp, jacr, nv);
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment jpSeg = jacp != null ? temp.allocate(ValueLayout.JAVA_DOUBLE, 3L * nv) : MemorySegment.NULL;
            MemorySegment jrSeg = jacr != null ? temp.allocate(ValueLayout.JAVA_DOUBLE, 3L * nv) : MemorySegment.NULL;
            mujoco_h.mj_jacSite(m.segment(), d.segment(), jpSeg, jrSeg, site);
            copyOutJac(jpSeg, jacp, jrSeg, jacr, 3 * nv);
        }
    }

    public static void jacPointAxis(MjModel m, MjData d, double[] jacPoint, double[] jacAxis, double[] point, double[] axis, int body) {
        checkArgs(m, d);
        if (point == null || point.length < 3 || axis == null || axis.length < 3) {
            throw new IllegalArgumentException("point and axis must have length >= 3");
        }
        int nv = m.nv();
        checkJacBuffers(jacPoint, jacAxis, nv);
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment jpSeg = jacPoint != null ? temp.allocate(ValueLayout.JAVA_DOUBLE, 3L * nv) : MemorySegment.NULL;
            MemorySegment jaSeg = jacAxis != null ? temp.allocate(ValueLayout.JAVA_DOUBLE, 3L * nv) : MemorySegment.NULL;
            MemorySegment ptSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, point);
            MemorySegment axSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, axis);
            mujoco_h.mj_jacPointAxis(m.segment(), d.segment(), jpSeg, jaSeg, ptSeg, axSeg, body);
            copyOutJac(jpSeg, jacPoint, jaSeg, jacAxis, 3 * nv);
        }
    }

    public static void jacDot(MjModel m, MjData d, double[] jacp, double[] jacr, double[] point, int body) {
        checkArgs(m, d);
        if (point == null || point.length < 3) {
            throw new IllegalArgumentException("point must have length >= 3");
        }
        int nv = m.nv();
        checkJacBuffers(jacp, jacr, nv);
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment jpSeg = jacp != null ? temp.allocate(ValueLayout.JAVA_DOUBLE, 3L * nv) : MemorySegment.NULL;
            MemorySegment jrSeg = jacr != null ? temp.allocate(ValueLayout.JAVA_DOUBLE, 3L * nv) : MemorySegment.NULL;
            MemorySegment ptSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, point);
            mujoco_h.mj_jacDot(m.segment(), d.segment(), jpSeg, jrSeg, ptSeg, body);
            copyOutJac(jpSeg, jacp, jrSeg, jacr, 3 * nv);
        }
    }

    public static void angmomMat(MjModel m, MjData d, double[] mat, int body) {
        checkArgs(m, d);
        int nv = m.nv();
        if (mat == null || mat.length < 3 * nv) {
            throw new IllegalArgumentException("mat length must be >= 3*nv (" + (3 * nv) + ")");
        }
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment matSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, 3L * nv);
            mujoco_h.mj_angmomMat(m.segment(), d.segment(), matSeg, body);
            MemorySegment.copy(matSeg, ValueLayout.JAVA_DOUBLE, 0, mat, 0, 3 * nv);
        }
    }

    /**
     * Multiplies constraint Jacobian {@code J} by vector {@code vec}: {@code res = J * vec} (`mj_mulJacVec`).
     */
    public static void mulJacVec(MjModel m, MjData d, double[] res, double[] vec) {
        checkArgs(m, d);
        int nefc = d.nefc();
        int nv = m.nv();
        if (res == null || res.length < nefc || vec == null || vec.length < nv) {
            throw new IllegalArgumentException("res length must be >= nefc (" + nefc + ") and vec length >= nv (" + nv + ")");
        }
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment resSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, Math.max(1, nefc));
            MemorySegment vecSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vec);
            mujoco_h.mj_mulJacVec(m.segment(), d.segment(), resSeg, vecSeg);
            if (nefc > 0) {
                MemorySegment.copy(resSeg, ValueLayout.JAVA_DOUBLE, 0, res, 0, nefc);
            }
        }
    }

    /**
     * Multiplies transposed constraint Jacobian {@code J^T} by vector {@code vec}: {@code res = J^T * vec} (`mj_mulJacTVec`).
     */
    public static void mulJacTVec(MjModel m, MjData d, double[] res, double[] vec) {
        checkArgs(m, d);
        int nefc = d.nefc();
        int nv = m.nv();
        if (res == null || res.length < nv || vec == null || vec.length < nefc) {
            throw new IllegalArgumentException("res length must be >= nv (" + nv + ") and vec length >= nefc (" + nefc + ")");
        }
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment resSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, Math.max(1, nv));
            MemorySegment vecSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vec);
            mujoco_h.mj_mulJacTVec(m.segment(), d.segment(), resSeg, vecSeg);
            if (nv > 0) {
                MemorySegment.copy(resSeg, ValueLayout.JAVA_DOUBLE, 0, res, 0, nv);
            }
        }
    }

    // =========================================================================
    // Forces, Velocities, Position Integration & Differences
    // =========================================================================

    /**
     * Applies Cartesian force (3) and torque (3) at {@code point} on {@code body}, accumulating into {@code qfrcTarget} (`mj_applyFT`).
     */
    public static void applyFT(MjModel m, MjData d, double[] force, double[] torque, double[] point, int body, double[] qfrcTarget) {
        checkArgs(m, d);
        int nv = m.nv();
        if (force == null || force.length < 3 || torque == null || torque.length < 3 || point == null || point.length < 3) {
            throw new IllegalArgumentException("force, torque, and point must have length >= 3");
        }
        if (qfrcTarget == null || qfrcTarget.length < nv) {
            throw new IllegalArgumentException("qfrcTarget must have length >= nv (" + nv + ")");
        }
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment fSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, force);
            MemorySegment tSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, torque);
            MemorySegment pSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, point);
            MemorySegment qSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, qfrcTarget);
            mujoco_h.mj_applyFT(m.segment(), d.segment(), fSeg, tSeg, pSeg, body, qSeg);
            MemorySegment.copy(qSeg, ValueLayout.JAVA_DOUBLE, 0, qfrcTarget, 0, nv);
        }
    }

    /**
     * Computes 6D object velocity (3 rotational, 3 translational) (`mj_objectVelocity`).
     */
    public static void objectVelocity(MjModel m, MjData d, MjObjType objtype, int objid, double[] res, int flgLocal) {
        Objects.requireNonNull(objtype, "objtype must not be null");
        objectVelocity(m, d, objtype.code(), objid, res, flgLocal);
    }

    public static void objectVelocity(MjModel m, MjData d, int objtype, int objid, double[] res, int flgLocal) {
        checkArgs(m, d);
        if (res == null || res.length < 6) {
            throw new IllegalArgumentException("res must have length >= 6");
        }
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment resSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, 6);
            mujoco_h.mj_objectVelocity(m.segment(), d.segment(), objtype, objid, resSeg, flgLocal);
            MemorySegment.copy(resSeg, ValueLayout.JAVA_DOUBLE, 0, res, 0, 6);
        }
    }

    /**
     * Computes 6D object acceleration (3 rotational, 3 translational) (`mj_objectAcceleration`).
     */
    public static void objectAcceleration(MjModel m, MjData d, MjObjType objtype, int objid, double[] res, int flgLocal) {
        Objects.requireNonNull(objtype, "objtype must not be null");
        objectAcceleration(m, d, objtype.code(), objid, res, flgLocal);
    }

    public static void objectAcceleration(MjModel m, MjData d, int objtype, int objid, double[] res, int flgLocal) {
        checkArgs(m, d);
        if (res == null || res.length < 6) {
            throw new IllegalArgumentException("res must have length >= 6");
        }
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment resSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, 6);
            mujoco_h.mj_objectAcceleration(m.segment(), d.segment(), objtype, objid, resSeg, flgLocal);
            MemorySegment.copy(resSeg, ValueLayout.JAVA_DOUBLE, 0, res, 0, 6);
        }
    }

    /**
     * Extracts 6D contact force/torque for contact {@code id} in contact frame (`mj_contactForce`).
     */
    public static void contactForce(MjModel m, MjData d, int id, double[] result) {
        checkArgs(m, d);
        if (result == null || result.length < 6) {
            throw new IllegalArgumentException("result must have length >= 6");
        }
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment resSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, 6);
            mujoco_h.mj_contactForce(m.segment(), d.segment(), id, resSeg);
            MemorySegment.copy(resSeg, ValueLayout.JAVA_DOUBLE, 0, result, 0, 6);
        }
    }

    public static double[] contactForce(MjModel m, MjData d, int id) {
        double[] res = new double[6];
        contactForce(m, d, id, res);
        return res;
    }

    /**
     * Computes velocity by finite-differencing two positions: {@code qvel = (qpos2 - qpos1) / dt} (`mj_differentiatePos`).
     */
    public static void differentiatePos(MjModel m, double[] qvel, double dt, double[] qpos1, double[] qpos2) {
        Objects.requireNonNull(m, "m must not be null");
        m.checkOpen();
        int nq = m.nq();
        int nv = m.nv();
        if (qvel == null || qvel.length < nv || qpos1 == null || qpos1.length < nq || qpos2 == null || qpos2.length < nq) {
            throw new IllegalArgumentException("qvel must have length >= nv (" + nv + ") and qpos1/qpos2 >= nq (" + nq + ")");
        }
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment qvSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, nv);
            MemorySegment qp1Seg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, qpos1);
            MemorySegment qp2Seg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, qpos2);
            mujoco_h.mj_differentiatePos(m.segment(), qvSeg, dt, qp1Seg, qp2Seg);
            MemorySegment.copy(qvSeg, ValueLayout.JAVA_DOUBLE, 0, qvel, 0, nv);
        }
    }

    /**
     * Integrates position {@code qpos} in-place given velocity {@code qvel} and step {@code dt} (`mj_integratePos`).
     */
    public static void integratePos(MjModel m, double[] qpos, double[] qvel, double dt) {
        Objects.requireNonNull(m, "m must not be null");
        m.checkOpen();
        int nq = m.nq();
        int nv = m.nv();
        if (qpos == null || qpos.length < nq || qvel == null || qvel.length < nv) {
            throw new IllegalArgumentException("qpos must have length >= nq (" + nq + ") and qvel >= nv (" + nv + ")");
        }
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment qpSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, qpos);
            MemorySegment qvSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, qvel);
            mujoco_h.mj_integratePos(m.segment(), qpSeg, qvSeg, dt);
            MemorySegment.copy(qpSeg, ValueLayout.JAVA_DOUBLE, 0, qpos, 0, nq);
        }
    }

    /**
     * Normalizes all quaternions inside {@code qpos} in-place (`mj_normalizeQuat`).
     */
    public static void normalizeQuat(MjModel m, double[] qpos) {
        Objects.requireNonNull(m, "m must not be null");
        m.checkOpen();
        int nq = m.nq();
        if (qpos == null || qpos.length < nq) {
            throw new IllegalArgumentException("qpos must have length >= nq (" + nq + ")");
        }
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment qpSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, qpos);
            mujoco_h.mj_normalizeQuat(m.segment(), qpSeg);
            MemorySegment.copy(qpSeg, ValueLayout.JAVA_DOUBLE, 0, qpos, 0, nq);
        }
    }

    /**
     * Normalizes all quaternions inside {@code d->qpos} in-place (`mj_normalizeQuat`).
     */
    public static void normalizeQuat(MjModel m, MjData d) {
        checkArgs(m, d);
        mujoco_h.mj_normalizeQuat(m.segment(), d.qpos().segment());
    }

    /**
     * Normalizes all quaternions inside {@code qpos} vector in-place (`mj_normalizeQuat`).
     */
    public static void normalizeQuat(MjModel m, MjDoubleVector qpos) {
        Objects.requireNonNull(m, "m must not be null");
        Objects.requireNonNull(qpos, "qpos must not be null");
        m.checkOpen();
        qpos.checkOpen();
        if (qpos.size() < m.nq()) {
            throw new IllegalArgumentException("qpos size must be >= nq (" + m.nq() + ")");
        }
        mujoco_h.mj_normalizeQuat(m.segment(), qpos.segment());
    }

    /**
     * Maps local body coordinates {@code pos} and {@code quat} to global coordinates {@code xpos} and {@code xmat} (`mj_local2Global`).
     */
    public static void local2Global(MjData d, double[] xpos, double[] xmat, double[] pos, double[] quat, int body, byte sameframe) {
        Objects.requireNonNull(d, "d must not be null");
        d.checkOpen();
        if (xpos == null || xpos.length < 3 || xmat == null || xmat.length < 9 || pos == null || pos.length < 3 || quat == null || quat.length < 4) {
            throw new IllegalArgumentException("Invalid array lengths for local2Global");
        }
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment xpSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, 3);
            MemorySegment xmSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, 9);
            MemorySegment pSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, pos);
            MemorySegment qSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, quat);
            mujoco_h.mj_local2Global(d.segment(), xpSeg, xmSeg, pSeg, qSeg, body, sameframe);
            MemorySegment.copy(xpSeg, ValueLayout.JAVA_DOUBLE, 0, xpos, 0, 3);
            MemorySegment.copy(xmSeg, ValueLayout.JAVA_DOUBLE, 0, xmat, 0, 9);
        }
    }

    /**
     * Computes signed distance between two geoms (`mj_geomDistance`).
     */
    public static double geomDistance(MjModel m, MjData d, int geom1, int geom2, double distmax, double[] fromto) {
        checkArgs(m, d);
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment ftSeg = fromto != null && fromto.length >= 6
                    ? temp.allocate(ValueLayout.JAVA_DOUBLE, 6)
                    : MemorySegment.NULL;
            double dist = mujoco_h.mj_geomDistance(m.segment(), d.segment(), geom1, geom2, distmax, ftSeg);
            if (!ftSeg.equals(MemorySegment.NULL)) {
                MemorySegment.copy(ftSeg, ValueLayout.JAVA_DOUBLE, 0, fromto, 0, 6);
            }
            return dist;
        }
    }

    // =========================================================================
    // Delay & History Buffers (mj_readCtrl, mj_readSensor, mj_initCtrlHistory, mj_initSensorHistory)
    // =========================================================================

    /**
     * Reads delayed actuator control value (`mj_readCtrl`).
     * Handles both {@code interp == 0} (where MuJoCo returns a pointer into {@code d->history})
     * and {@code interp != 0} (where MuJoCo writes into the result buffer).
     */
    public static double readCtrl(MjModel m, MjData d, int id, double time, int interp) {
        checkArgs(m, d);
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment resSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, 1);
            MemorySegment retPtr = mujoco_h.mj_readCtrl(m.segment(), d.segment(), id, time, resSeg, interp);
            if (!retPtr.equals(MemorySegment.NULL)) {
                return retPtr.reinterpret(Double.BYTES).get(ValueLayout.JAVA_DOUBLE, 0);
            }
            return resSeg.get(ValueLayout.JAVA_DOUBLE, 0);
        }
    }

    /**
     * Reads delayed actuator control values (`mj_readCtrl`) into {@code result}.
     */
    public static void readCtrl(MjModel m, MjData d, int id, double time, double[] result, int interp) {
        checkArgs(m, d);
        if (id < 0 || id >= m.nactuator()) {
            throw new IllegalArgumentException("Invalid actuator id: " + id);
        }
        int dim = Math.max(1, m.actuatorCtrlnum().get(id));
        if (result == null || result.length < dim) {
            throw new IllegalArgumentException("result length must be >= actuator ctrlnum (" + dim + ")");
        }
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment resSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, dim);
            MemorySegment retPtr = mujoco_h.mj_readCtrl(m.segment(), d.segment(), id, time, resSeg, interp);
            if (!retPtr.equals(MemorySegment.NULL)) {
                MemorySegment.copy(retPtr.reinterpret((long) dim * Double.BYTES), ValueLayout.JAVA_DOUBLE, 0, result, 0, dim);
            } else {
                MemorySegment.copy(resSeg, ValueLayout.JAVA_DOUBLE, 0, result, 0, dim);
            }
        }
    }

    /**
     * Returns true if the 3D world {@code point} is inside site {@code siteid} (`mj_insideSite`).
     */
    public static boolean insideSite(MjModel m, MjData d, int siteid, double[] point) {
        checkArgs(m, d);
        if (point == null || point.length < 3) {
            throw new IllegalArgumentException("point must have length >= 3");
        }
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment ptSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, point);
            return mujoco_h.mj_insideSite(m.segment(), d.segment(), siteid, ptSeg) != 0;
        }
    }

    /**
     * Prints the MJCF XML schema to a Java {@link String} (`mj_printSchema`).
     */
    public static String printSchema(boolean flgHtml, boolean flgPad) {
        MjNativeLoader.ensureLoaded();
        int bufSize = 131072;
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment buf = temp.allocate(bufSize);
            mujoco_h.mj_printSchema(MemorySegment.NULL, buf, bufSize, flgHtml ? 1 : 0, flgPad ? 1 : 0);
            return buf.getString(0);
        }
    }

    /**
     * Reads delayed sensor data (`mj_readSensor`) into {@code result}.
     * Handles both {@code interp == 0} (pointer return) and {@code interp != 0} (buffer write).
     */
    public static void readSensor(MjModel m, MjData d, int id, double time, double[] result, int interp) {
        checkArgs(m, d);
        if (id < 0 || id >= m.nsensor()) {
            throw new IllegalArgumentException("Invalid sensor id: " + id);
        }
        int dim = m.sensorDim().get(id);
        if (result == null || result.length < dim) {
            throw new IllegalArgumentException("result length must be >= sensor dim (" + dim + ")");
        }
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment resSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, dim);
            MemorySegment retPtr = mujoco_h.mj_readSensor(m.segment(), d.segment(), id, time, resSeg, interp);
            if (!retPtr.equals(MemorySegment.NULL)) {
                MemorySegment.copy(retPtr.reinterpret((long) dim * Double.BYTES), ValueLayout.JAVA_DOUBLE, 0, result, 0, dim);
            } else {
                MemorySegment.copy(resSeg, ValueLayout.JAVA_DOUBLE, 0, result, 0, dim);
            }
        }
    }

    /**
     * Initializes actuator control history buffer (`mj_initCtrlHistory`).
     */
    public static void initCtrlHistory(MjModel m, MjData d, int id, double[] times, double[] values) {
        checkArgs(m, d);
        if (id < 0 || id >= m.nactuator()) {
            throw new IllegalArgumentException("Invalid actuator id: " + id);
        }
        Objects.requireNonNull(values, "values must not be null");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment tSeg = times != null ? temp.allocateFrom(ValueLayout.JAVA_DOUBLE, times) : MemorySegment.NULL;
            MemorySegment vSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, values);
            mujoco_h.mj_initCtrlHistory(m.segment(), d.segment(), id, tSeg, vSeg);
        }
    }

    /**
     * Initializes sensor history buffer (`mj_initSensorHistory`).
     */
    public static void initSensorHistory(MjModel m, MjData d, int id, double[] times, double[] values, double phase) {
        checkArgs(m, d);
        if (id < 0 || id >= m.nsensor()) {
            throw new IllegalArgumentException("Invalid sensor id: " + id);
        }
        Objects.requireNonNull(values, "values must not be null");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment tSeg = times != null ? temp.allocateFrom(ValueLayout.JAVA_DOUBLE, times) : MemorySegment.NULL;
            MemorySegment vSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, values);
            mujoco_h.mj_initSensorHistory(m.segment(), d.segment(), id, tSeg, vSeg, phase);
        }
    }

    // =========================================================================
    // Finite-Difference Derivatives (mjd_transitionFD, mjd_inverseFD, mjd_quatIntegrate, mjd_subQuat)
    // =========================================================================

    /**
     * Finite-difference transition matrices {@code A} (`dx_{t+1}/dx_t`) and {@code B} (`dx_{t+1}/du_t`)
     * and sensor derivatives {@code C} (`dy_t/dx_t`) and {@code D} (`dy_t/du_t`) (`mjd_transitionFD`).
     * Any of {@code A}, {@code B}, {@code C}, {@code D} may be {@code null}.
     */
    public static void transitionFD(
            MjModel m, MjData d, double eps, boolean flgCentered,
            double[] A, double[] B, double[] C, double[] D) {
        checkArgs(m, d);
        int ndx = 2 * m.nv() + m.na();
        int nu = m.nu();
        int ns = m.nsensordata();
        if (A != null && A.length < ndx * ndx) throw new IllegalArgumentException("A length must be >= " + (ndx * ndx));
        if (B != null && B.length < ndx * nu) throw new IllegalArgumentException("B length must be >= " + (ndx * nu));
        if (C != null && C.length < ns * ndx) throw new IllegalArgumentException("C length must be >= " + (ns * ndx));
        if (D != null && D.length < ns * nu) throw new IllegalArgumentException("D length must be >= " + (ns * nu));

        try (Arena temp = Arena.ofConfined()) {
            MemorySegment aSeg = A != null ? temp.allocate(ValueLayout.JAVA_DOUBLE, Math.max(1L, (long) ndx * ndx)) : MemorySegment.NULL;
            MemorySegment bSeg = B != null ? temp.allocate(ValueLayout.JAVA_DOUBLE, Math.max(1L, (long) ndx * nu)) : MemorySegment.NULL;
            MemorySegment cSeg = C != null ? temp.allocate(ValueLayout.JAVA_DOUBLE, Math.max(1L, (long) ns * ndx)) : MemorySegment.NULL;
            MemorySegment dSeg = D != null ? temp.allocate(ValueLayout.JAVA_DOUBLE, Math.max(1L, (long) ns * nu)) : MemorySegment.NULL;
            mujoco_h.mjd_transitionFD(m.segment(), d.segment(), eps, flgCentered, aSeg, bSeg, cSeg, dSeg);
            if (A != null && ndx * ndx > 0) MemorySegment.copy(aSeg, ValueLayout.JAVA_DOUBLE, 0, A, 0, ndx * ndx);
            if (B != null && ndx * nu > 0) MemorySegment.copy(bSeg, ValueLayout.JAVA_DOUBLE, 0, B, 0, ndx * nu);
            if (C != null && ns * ndx > 0) MemorySegment.copy(cSeg, ValueLayout.JAVA_DOUBLE, 0, C, 0, ns * ndx);
            if (D != null && ns * nu > 0) MemorySegment.copy(dSeg, ValueLayout.JAVA_DOUBLE, 0, D, 0, ns * nu);
        }
    }

    /**
     * Finite-difference inverse dynamics derivatives (`mjd_inverseFD`).
     * Any of {@code DfDq}, {@code DfDv}, {@code DfDa}, {@code DsDq}, {@code DsDv}, {@code DsDa}, {@code DmDq} may be {@code null}.
     */
    public static void inverseFD(
            MjModel m, MjData d, double eps, boolean flgActuation,
            double[] DfDq, double[] DfDv, double[] DfDa,
            double[] DsDq, double[] DsDv, double[] DsDa,
            double[] DmDq) {
        checkArgs(m, d);
        int nv = m.nv();
        int ns = m.nsensordata();
        int nM = m.nM();
        int nv2 = nv * nv;
        int nsv = ns * nv;
        int nMv = nM * nv;

        if (DfDq != null && DfDq.length < nv2) throw new IllegalArgumentException("DfDq length < " + nv2);
        if (DfDv != null && DfDv.length < nv2) throw new IllegalArgumentException("DfDv length < " + nv2);
        if (DfDa != null && DfDa.length < nv2) throw new IllegalArgumentException("DfDa length < " + nv2);
        if (DsDq != null && DsDq.length < nsv) throw new IllegalArgumentException("DsDq length < " + nsv);
        if (DsDv != null && DsDv.length < nsv) throw new IllegalArgumentException("DsDv length < " + nsv);
        if (DsDa != null && DsDa.length < nsv) throw new IllegalArgumentException("DsDa length < " + nsv);
        if (DmDq != null && DmDq.length < nMv) throw new IllegalArgumentException("DmDq length < " + nMv);

        try (Arena temp = Arena.ofConfined()) {
            MemorySegment fqSeg = DfDq != null ? temp.allocate(ValueLayout.JAVA_DOUBLE, Math.max(1, nv2)) : MemorySegment.NULL;
            MemorySegment fvSeg = DfDv != null ? temp.allocate(ValueLayout.JAVA_DOUBLE, Math.max(1, nv2)) : MemorySegment.NULL;
            MemorySegment faSeg = DfDa != null ? temp.allocate(ValueLayout.JAVA_DOUBLE, Math.max(1, nv2)) : MemorySegment.NULL;
            MemorySegment sqSeg = DsDq != null ? temp.allocate(ValueLayout.JAVA_DOUBLE, Math.max(1, nsv)) : MemorySegment.NULL;
            MemorySegment svSeg = DsDv != null ? temp.allocate(ValueLayout.JAVA_DOUBLE, Math.max(1, nsv)) : MemorySegment.NULL;
            MemorySegment saSeg = DsDa != null ? temp.allocate(ValueLayout.JAVA_DOUBLE, Math.max(1, nsv)) : MemorySegment.NULL;
            MemorySegment mqSeg = DmDq != null ? temp.allocate(ValueLayout.JAVA_DOUBLE, Math.max(1, nMv)) : MemorySegment.NULL;

            mujoco_h.mjd_inverseFD(m.segment(), d.segment(), eps, flgActuation, fqSeg, fvSeg, faSeg, sqSeg, svSeg, saSeg, mqSeg);

            if (DfDq != null && nv2 > 0) MemorySegment.copy(fqSeg, ValueLayout.JAVA_DOUBLE, 0, DfDq, 0, nv2);
            if (DfDv != null && nv2 > 0) MemorySegment.copy(fvSeg, ValueLayout.JAVA_DOUBLE, 0, DfDv, 0, nv2);
            if (DfDa != null && nv2 > 0) MemorySegment.copy(faSeg, ValueLayout.JAVA_DOUBLE, 0, DfDa, 0, nv2);
            if (DsDq != null && nsv > 0) MemorySegment.copy(sqSeg, ValueLayout.JAVA_DOUBLE, 0, DsDq, 0, nsv);
            if (DsDv != null && nsv > 0) MemorySegment.copy(svSeg, ValueLayout.JAVA_DOUBLE, 0, DsDv, 0, nsv);
            if (DsDa != null && nsv > 0) MemorySegment.copy(saSeg, ValueLayout.JAVA_DOUBLE, 0, DsDa, 0, nsv);
            if (DmDq != null && nMv > 0) MemorySegment.copy(mqSeg, ValueLayout.JAVA_DOUBLE, 0, DmDq, 0, nMv);
        }
    }

    /**
     * Derivatives of {@code mju_subQuat(qa, qb)} with respect to {@code qa} and {@code qb} (`mjd_subQuat`).
     */
    public static void subQuatFD(double[] qa, double[] qb, double[] Da, double[] Db) {
        if (qa == null || qa.length < 4 || qb == null || qb.length < 4) {
            throw new IllegalArgumentException("qa and qb must have length >= 4");
        }
        if (Da != null && Da.length < 9) throw new IllegalArgumentException("Da length < 9");
        if (Db != null && Db.length < 9) throw new IllegalArgumentException("Db length < 9");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment qaSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, qa);
            MemorySegment qbSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, qb);
            MemorySegment daSeg = Da != null ? temp.allocate(ValueLayout.JAVA_DOUBLE, 9) : MemorySegment.NULL;
            MemorySegment dbSeg = Db != null ? temp.allocate(ValueLayout.JAVA_DOUBLE, 9) : MemorySegment.NULL;
            mujoco_h.mjd_subQuat(qaSeg, qbSeg, daSeg, dbSeg);
            if (Da != null) MemorySegment.copy(daSeg, ValueLayout.JAVA_DOUBLE, 0, Da, 0, 9);
            if (Db != null) MemorySegment.copy(dbSeg, ValueLayout.JAVA_DOUBLE, 0, Db, 0, 9);
        }
    }

    /**
     * Derivatives of {@code mju_quatIntegrate} (`mjd_quatIntegrate`).
     */
    public static void quatIntegrateFD(double[] vel, double scale, double[] Dq, double[] Dv, double[] Dh) {
        if (vel == null || vel.length < 3) {
            throw new IllegalArgumentException("vel must have length >= 3");
        }
        if (Dq == null || Dq.length < 9 || Dv == null || Dv.length < 9 || Dh == null || Dh.length < 3) {
            throw new IllegalArgumentException("Dq (9), Dv (9), and Dh (3) must be non-null with sufficient length");
        }
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment vSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vel);
            MemorySegment dqSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, 9);
            MemorySegment dvSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, 9);
            MemorySegment dhSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, 3);
            mujoco_h.mjd_quatIntegrate(vSeg, scale, dqSeg, dvSeg, dhSeg);
            MemorySegment.copy(dqSeg, ValueLayout.JAVA_DOUBLE, 0, Dq, 0, 9);
            MemorySegment.copy(dvSeg, ValueLayout.JAVA_DOUBLE, 0, Dv, 0, 9);
            MemorySegment.copy(dhSeg, ValueLayout.JAVA_DOUBLE, 0, Dh, 0, 3);
        }
    }

    // =========================================================================
    // Internal Helpers
    // =========================================================================

    private static void checkArgs(MjModel m, MjData d) {
        Objects.requireNonNull(m, "m must not be null");
        Objects.requireNonNull(d, "d must not be null");
        m.checkOpen();
        d.checkOpen();
    }

    private static void checkJacBuffers(double[] jacp, double[] jacr, int nv) {
        int req = 3 * nv;
        if (jacp != null && jacp.length < req) {
            throw new IllegalArgumentException("jacp length (" + jacp.length + ") < 3*nv (" + req + ")");
        }
        if (jacr != null && jacr.length < req) {
            throw new IllegalArgumentException("jacr length (" + jacr.length + ") < 3*nv (" + req + ")");
        }
    }

    private static void copyOutJac(MemorySegment jpSeg, double[] jacp, MemorySegment jrSeg, double[] jacr, int len) {
        if (len <= 0) return;
        if (jacp != null) {
            MemorySegment.copy(jpSeg, ValueLayout.JAVA_DOUBLE, 0, jacp, 0, len);
        }
        if (jacr != null) {
            MemorySegment.copy(jrSeg, ValueLayout.JAVA_DOUBLE, 0, jacr, 0, len);
        }
    }
}
