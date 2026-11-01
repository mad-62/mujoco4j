package org.mujoco4j;

import org.mujoco4j.internal.panama.mujoco_h;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Objects;

/**
 * High-level mathematical, quaternion, matrix, decomposition, QP solver, and utility
 * functions wrapping MuJoCo's {@code mju_*} C functions.
 */
public final class MjMath {

    static {
        MjNativeLoader.ensureLoaded();
    }

    private MjMath() {}

    // =========================================================================
    // 3D Vector Operations
    // =========================================================================

    public static void zero3(double[] res) {
        checkLen(res, 3, "res");
        res[0] = 0.0;
        res[1] = 0.0;
        res[2] = 0.0;
    }

    public static void copy3(double[] res, double[] data) {
        checkLen(res, 3, "res");
        checkLen(data, 3, "data");
        System.arraycopy(data, 0, res, 0, 3);
    }

    public static void scl3(double[] res, double[] vec, double scl) {
        checkLen(res, 3, "res");
        checkLen(vec, 3, "vec");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment rSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, 3);
            MemorySegment vSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vec);
            mujoco_h.mju_scl3(rSeg, vSeg, scl);
            MemorySegment.copy(rSeg, ValueLayout.JAVA_DOUBLE, 0, res, 0, 3);
        }
    }

    public static void add3(double[] res, double[] vec1, double[] vec2) {
        checkLen(res, 3, "res");
        checkLen(vec1, 3, "vec1");
        checkLen(vec2, 3, "vec2");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment rSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, 3);
            MemorySegment v1Seg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vec1);
            MemorySegment v2Seg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vec2);
            mujoco_h.mju_add3(rSeg, v1Seg, v2Seg);
            MemorySegment.copy(rSeg, ValueLayout.JAVA_DOUBLE, 0, res, 0, 3);
        }
    }

    public static void sub3(double[] res, double[] vec1, double[] vec2) {
        checkLen(res, 3, "res");
        checkLen(vec1, 3, "vec1");
        checkLen(vec2, 3, "vec2");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment rSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, 3);
            MemorySegment v1Seg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vec1);
            MemorySegment v2Seg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vec2);
            mujoco_h.mju_sub3(rSeg, v1Seg, v2Seg);
            MemorySegment.copy(rSeg, ValueLayout.JAVA_DOUBLE, 0, res, 0, 3);
        }
    }

    public static void addTo3(double[] res, double[] vec) {
        checkLen(res, 3, "res");
        checkLen(vec, 3, "vec");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment rSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, res);
            MemorySegment vSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vec);
            mujoco_h.mju_addTo3(rSeg, vSeg);
            MemorySegment.copy(rSeg, ValueLayout.JAVA_DOUBLE, 0, res, 0, 3);
        }
    }

    public static void subFrom3(double[] res, double[] vec) {
        checkLen(res, 3, "res");
        checkLen(vec, 3, "vec");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment rSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, res);
            MemorySegment vSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vec);
            mujoco_h.mju_subFrom3(rSeg, vSeg);
            MemorySegment.copy(rSeg, ValueLayout.JAVA_DOUBLE, 0, res, 0, 3);
        }
    }

    public static void addToScl3(double[] res, double[] vec, double scl) {
        checkLen(res, 3, "res");
        checkLen(vec, 3, "vec");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment rSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, res);
            MemorySegment vSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vec);
            mujoco_h.mju_addToScl3(rSeg, vSeg, scl);
            MemorySegment.copy(rSeg, ValueLayout.JAVA_DOUBLE, 0, res, 0, 3);
        }
    }

    public static void addScl3(double[] res, double[] vec1, double[] vec2, double scl) {
        checkLen(res, 3, "res");
        checkLen(vec1, 3, "vec1");
        checkLen(vec2, 3, "vec2");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment rSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, 3);
            MemorySegment v1Seg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vec1);
            MemorySegment v2Seg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vec2);
            mujoco_h.mju_addScl3(rSeg, v1Seg, v2Seg, scl);
            MemorySegment.copy(rSeg, ValueLayout.JAVA_DOUBLE, 0, res, 0, 3);
        }
    }

    public static double normalize3(double[] vec) {
        checkLen(vec, 3, "vec");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment vSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vec);
            double norm = mujoco_h.mju_normalize3(vSeg);
            MemorySegment.copy(vSeg, ValueLayout.JAVA_DOUBLE, 0, vec, 0, 3);
            return norm;
        }
    }

    public static double norm3(double[] vec) {
        checkLen(vec, 3, "vec");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment vSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vec);
            return mujoco_h.mju_norm3(vSeg);
        }
    }

    public static double dot3(double[] vec1, double[] vec2) {
        checkLen(vec1, 3, "vec1");
        checkLen(vec2, 3, "vec2");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment v1Seg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vec1);
            MemorySegment v2Seg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vec2);
            return mujoco_h.mju_dot3(v1Seg, v2Seg);
        }
    }

    public static double dist3(double[] pos1, double[] pos2) {
        checkLen(pos1, 3, "pos1");
        checkLen(pos2, 3, "pos2");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment p1Seg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, pos1);
            MemorySegment p2Seg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, pos2);
            return mujoco_h.mju_dist3(p1Seg, p2Seg);
        }
    }

    public static void mulMatVec3(double[] res, double[] mat, double[] vec) {
        checkLen(res, 3, "res");
        checkLen(mat, 9, "mat");
        checkLen(vec, 3, "vec");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment rSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, 3);
            MemorySegment mSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, mat);
            MemorySegment vSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vec);
            mujoco_h.mju_mulMatVec3(rSeg, mSeg, vSeg);
            MemorySegment.copy(rSeg, ValueLayout.JAVA_DOUBLE, 0, res, 0, 3);
        }
    }

    public static void mulMatTVec3(double[] res, double[] mat, double[] vec) {
        checkLen(res, 3, "res");
        checkLen(mat, 9, "mat");
        checkLen(vec, 3, "vec");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment rSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, 3);
            MemorySegment mSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, mat);
            MemorySegment vSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vec);
            mujoco_h.mju_mulMatTVec3(rSeg, mSeg, vSeg);
            MemorySegment.copy(rSeg, ValueLayout.JAVA_DOUBLE, 0, res, 0, 3);
        }
    }

    public static void cross(double[] res, double[] a, double[] b) {
        checkLen(res, 3, "res");
        checkLen(a, 3, "a");
        checkLen(b, 3, "b");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment rSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, 3);
            MemorySegment aSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, a);
            MemorySegment bSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, b);
            mujoco_h.mju_cross(rSeg, aSeg, bSeg);
            MemorySegment.copy(rSeg, ValueLayout.JAVA_DOUBLE, 0, res, 0, 3);
        }
    }

    // =========================================================================
    // 4D Vector & Quaternion Operations
    // =========================================================================

    public static void zero4(double[] res) {
        checkLen(res, 4, "res");
        res[0] = 0.0;
        res[1] = 0.0;
        res[2] = 0.0;
        res[3] = 0.0;
    }

    public static void unit4(double[] res) {
        checkLen(res, 4, "res");
        res[0] = 1.0;
        res[1] = 0.0;
        res[2] = 0.0;
        res[3] = 0.0;
    }

    public static void copy4(double[] res, double[] data) {
        checkLen(res, 4, "res");
        checkLen(data, 4, "data");
        System.arraycopy(data, 0, res, 0, 4);
    }

    public static double normalize4(double[] vec) {
        checkLen(vec, 4, "vec");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment vSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vec);
            double norm = mujoco_h.mju_normalize4(vSeg);
            MemorySegment.copy(vSeg, ValueLayout.JAVA_DOUBLE, 0, vec, 0, 4);
            return norm;
        }
    }

    public static void rotVecQuat(double[] res, double[] vec, double[] quat) {
        checkLen(res, 3, "res");
        checkLen(vec, 3, "vec");
        checkLen(quat, 4, "quat");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment rSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, 3);
            MemorySegment vSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vec);
            MemorySegment qSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, quat);
            mujoco_h.mju_rotVecQuat(rSeg, vSeg, qSeg);
            MemorySegment.copy(rSeg, ValueLayout.JAVA_DOUBLE, 0, res, 0, 3);
        }
    }

    public static void negQuat(double[] res, double[] quat) {
        checkLen(res, 4, "res");
        checkLen(quat, 4, "quat");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment rSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, 4);
            MemorySegment qSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, quat);
            mujoco_h.mju_negQuat(rSeg, qSeg);
            MemorySegment.copy(rSeg, ValueLayout.JAVA_DOUBLE, 0, res, 0, 4);
        }
    }

    public static void mulQuat(double[] res, double[] quat1, double[] quat2) {
        checkLen(res, 4, "res");
        checkLen(quat1, 4, "quat1");
        checkLen(quat2, 4, "quat2");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment rSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, 4);
            MemorySegment q1Seg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, quat1);
            MemorySegment q2Seg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, quat2);
            mujoco_h.mju_mulQuat(rSeg, q1Seg, q2Seg);
            MemorySegment.copy(rSeg, ValueLayout.JAVA_DOUBLE, 0, res, 0, 4);
        }
    }

    public static void mulQuatAxis(double[] res, double[] quat, double[] axis) {
        checkLen(res, 4, "res");
        checkLen(quat, 4, "quat");
        checkLen(axis, 3, "axis");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment rSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, 4);
            MemorySegment qSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, quat);
            MemorySegment aSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, axis);
            mujoco_h.mju_mulQuatAxis(rSeg, qSeg, aSeg);
            MemorySegment.copy(rSeg, ValueLayout.JAVA_DOUBLE, 0, res, 0, 4);
        }
    }

    public static void axisAngle2Quat(double[] res, double[] axis, double angle) {
        checkLen(res, 4, "res");
        checkLen(axis, 3, "axis");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment rSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, 4);
            MemorySegment aSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, axis);
            mujoco_h.mju_axisAngle2Quat(rSeg, aSeg, angle);
            MemorySegment.copy(rSeg, ValueLayout.JAVA_DOUBLE, 0, res, 0, 4);
        }
    }

    public static void quat2Vel(double[] res, double[] quat, double dt) {
        checkLen(res, 3, "res");
        checkLen(quat, 4, "quat");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment rSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, 3);
            MemorySegment qSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, quat);
            mujoco_h.mju_quat2Vel(rSeg, qSeg, dt);
            MemorySegment.copy(rSeg, ValueLayout.JAVA_DOUBLE, 0, res, 0, 3);
        }
    }

    public static void subQuat(double[] res, double[] qa, double[] qb) {
        checkLen(res, 3, "res");
        checkLen(qa, 4, "qa");
        checkLen(qb, 4, "qb");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment rSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, 3);
            MemorySegment qaSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, qa);
            MemorySegment qbSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, qb);
            mujoco_h.mju_subQuat(rSeg, qaSeg, qbSeg);
            MemorySegment.copy(rSeg, ValueLayout.JAVA_DOUBLE, 0, res, 0, 3);
        }
    }

    /**
     * Computes derivatives of {@code mju_subQuat(qa, qb)} with respect to {@code qa} and {@code qb} (`mjd_subQuat`).
     */
    public static void subQuat(double[] qa, double[] qb, double[] Da, double[] Db) {
        MjSim.subQuatFD(qa, qb, Da, Db);
    }

    public static void quat2Mat(double[] res, double[] quat) {
        checkLen(res, 9, "res");
        checkLen(quat, 4, "quat");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment rSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, 9);
            MemorySegment qSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, quat);
            mujoco_h.mju_quat2Mat(rSeg, qSeg);
            MemorySegment.copy(rSeg, ValueLayout.JAVA_DOUBLE, 0, res, 0, 9);
        }
    }

    public static void mat2Quat(double[] quat, double[] mat) {
        checkLen(quat, 4, "quat");
        checkLen(mat, 9, "mat");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment qSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, 4);
            MemorySegment mSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, mat);
            mujoco_h.mju_mat2Quat(qSeg, mSeg);
            MemorySegment.copy(qSeg, ValueLayout.JAVA_DOUBLE, 0, quat, 0, 4);
        }
    }

    public static void derivQuat(double[] res, double[] quat, double[] vel) {
        checkLen(res, 4, "res");
        checkLen(quat, 4, "quat");
        checkLen(vel, 3, "vel");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment rSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, 4);
            MemorySegment qSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, quat);
            MemorySegment vSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vel);
            mujoco_h.mju_derivQuat(rSeg, qSeg, vSeg);
            MemorySegment.copy(rSeg, ValueLayout.JAVA_DOUBLE, 0, res, 0, 4);
        }
    }

    public static void quatIntegrate(double[] quat, double[] vel, double scale) {
        checkLen(quat, 4, "quat");
        checkLen(vel, 3, "vel");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment qSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, quat);
            MemorySegment vSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vel);
            mujoco_h.mju_quatIntegrate(qSeg, vSeg, scale);
            MemorySegment.copy(qSeg, ValueLayout.JAVA_DOUBLE, 0, quat, 0, 4);
        }
    }

    public static void quatZ2Vec(double[] quat, double[] vec) {
        checkLen(quat, 4, "quat");
        checkLen(vec, 3, "vec");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment qSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, 4);
            MemorySegment vSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vec);
            mujoco_h.mju_quatZ2Vec(qSeg, vSeg);
            MemorySegment.copy(qSeg, ValueLayout.JAVA_DOUBLE, 0, quat, 0, 4);
        }
    }

    public static int mat2Rot(double[] quat, double[] mat) {
        checkLen(quat, 4, "quat");
        checkLen(mat, 9, "mat");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment qSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, quat);
            MemorySegment mSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, mat);
            int iter = mujoco_h.mju_mat2Rot(qSeg, mSeg);
            MemorySegment.copy(qSeg, ValueLayout.JAVA_DOUBLE, 0, quat, 0, 4);
            return iter;
        }
    }

    public static void euler2Quat(double[] quat, double[] euler, String seq) {
        checkLen(quat, 4, "quat");
        checkLen(euler, 3, "euler");
        Objects.requireNonNull(seq, "seq must not be null");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment qSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, 4);
            MemorySegment eSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, euler);
            MemorySegment sSeg = temp.allocateFrom(seq);
            mujoco_h.mju_euler2Quat(qSeg, eSeg, sSeg);
            MemorySegment.copy(qSeg, ValueLayout.JAVA_DOUBLE, 0, quat, 0, 4);
        }
    }

    // =========================================================================
    // Pose & Spatial Operations
    // =========================================================================

    public static void mulPose(double[] posres, double[] quatres, double[] pos1, double[] quat1, double[] pos2, double[] quat2) {
        checkLen(posres, 3, "posres");
        checkLen(quatres, 4, "quatres");
        checkLen(pos1, 3, "pos1");
        checkLen(quat1, 4, "quat1");
        checkLen(pos2, 3, "pos2");
        checkLen(quat2, 4, "quat2");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment prSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, 3);
            MemorySegment qrSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, 4);
            MemorySegment p1Seg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, pos1);
            MemorySegment q1Seg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, quat1);
            MemorySegment p2Seg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, pos2);
            MemorySegment q2Seg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, quat2);
            mujoco_h.mju_mulPose(prSeg, qrSeg, p1Seg, q1Seg, p2Seg, q2Seg);
            MemorySegment.copy(prSeg, ValueLayout.JAVA_DOUBLE, 0, posres, 0, 3);
            MemorySegment.copy(qrSeg, ValueLayout.JAVA_DOUBLE, 0, quatres, 0, 4);
        }
    }

    public static void negPose(double[] posres, double[] quatres, double[] pos, double[] quat) {
        checkLen(posres, 3, "posres");
        checkLen(quatres, 4, "quatres");
        checkLen(pos, 3, "pos");
        checkLen(quat, 4, "quat");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment prSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, 3);
            MemorySegment qrSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, 4);
            MemorySegment pSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, pos);
            MemorySegment qSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, quat);
            mujoco_h.mju_negPose(prSeg, qrSeg, pSeg, qSeg);
            MemorySegment.copy(prSeg, ValueLayout.JAVA_DOUBLE, 0, posres, 0, 3);
            MemorySegment.copy(qrSeg, ValueLayout.JAVA_DOUBLE, 0, quatres, 0, 4);
        }
    }

    public static void trnVecPose(double[] res, double[] pos, double[] quat, double[] vec) {
        checkLen(res, 3, "res");
        checkLen(pos, 3, "pos");
        checkLen(quat, 4, "quat");
        checkLen(vec, 3, "vec");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment rSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, 3);
            MemorySegment pSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, pos);
            MemorySegment qSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, quat);
            MemorySegment vSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vec);
            mujoco_h.mju_trnVecPose(rSeg, pSeg, qSeg, vSeg);
            MemorySegment.copy(rSeg, ValueLayout.JAVA_DOUBLE, 0, res, 0, 3);
        }
    }

    public static void transformSpatial(double[] res, double[] vec, int flgForce, double[] newpos, double[] oldpos, double[] rotnew2old) {
        checkLen(res, 6, "res");
        checkLen(vec, 6, "vec");
        checkLen(newpos, 3, "newpos");
        checkLen(oldpos, 3, "oldpos");
        checkLen(rotnew2old, 9, "rotnew2old");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment rSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, 6);
            MemorySegment vSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vec);
            MemorySegment npSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, newpos);
            MemorySegment opSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, oldpos);
            MemorySegment rotSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, rotnew2old);
            mujoco_h.mju_transformSpatial(rSeg, vSeg, flgForce, npSeg, opSeg, rotSeg);
            MemorySegment.copy(rSeg, ValueLayout.JAVA_DOUBLE, 0, res, 0, 6);
        }
    }

    // =========================================================================
    // N-Dimensional Vector & Matrix Operations
    // =========================================================================

    public static void zero(double[] res) {
        Objects.requireNonNull(res, "res must not be null");
        java.util.Arrays.fill(res, 0.0);
    }

    public static void fill(double[] res, double val) {
        Objects.requireNonNull(res, "res must not be null");
        java.util.Arrays.fill(res, val);
    }

    public static void copy(double[] res, double[] vec) {
        Objects.requireNonNull(res, "res must not be null");
        Objects.requireNonNull(vec, "vec must not be null");
        int n = Math.min(res.length, vec.length);
        System.arraycopy(vec, 0, res, 0, n);
    }

    public static double sum(double[] vec) {
        Objects.requireNonNull(vec, "vec must not be null");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment vSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vec);
            return mujoco_h.mju_sum(vSeg, vec.length);
        }
    }

    public static double L1(double[] vec) {
        Objects.requireNonNull(vec, "vec must not be null");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment vSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vec);
            return mujoco_h.mju_L1(vSeg, vec.length);
        }
    }

    public static void scl(double[] res, double[] vec, double scl) {
        Objects.requireNonNull(res, "res must not be null");
        Objects.requireNonNull(vec, "vec must not be null");
        int n = Math.min(res.length, vec.length);
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment rSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, n);
            MemorySegment vSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vec);
            mujoco_h.mju_scl(rSeg, vSeg, scl, n);
            MemorySegment.copy(rSeg, ValueLayout.JAVA_DOUBLE, 0, res, 0, n);
        }
    }

    public static void add(double[] res, double[] vec1, double[] vec2) {
        Objects.requireNonNull(res, "res must not be null");
        Objects.requireNonNull(vec1, "vec1 must not be null");
        Objects.requireNonNull(vec2, "vec2 must not be null");
        int n = Math.min(res.length, Math.min(vec1.length, vec2.length));
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment rSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, n);
            MemorySegment v1Seg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vec1);
            MemorySegment v2Seg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vec2);
            mujoco_h.mju_add(rSeg, v1Seg, v2Seg, n);
            MemorySegment.copy(rSeg, ValueLayout.JAVA_DOUBLE, 0, res, 0, n);
        }
    }

    public static void sub(double[] res, double[] vec1, double[] vec2) {
        Objects.requireNonNull(res, "res must not be null");
        Objects.requireNonNull(vec1, "vec1 must not be null");
        Objects.requireNonNull(vec2, "vec2 must not be null");
        int n = Math.min(res.length, Math.min(vec1.length, vec2.length));
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment rSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, n);
            MemorySegment v1Seg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vec1);
            MemorySegment v2Seg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vec2);
            mujoco_h.mju_sub(rSeg, v1Seg, v2Seg, n);
            MemorySegment.copy(rSeg, ValueLayout.JAVA_DOUBLE, 0, res, 0, n);
        }
    }

    public static void addTo(double[] res, double[] vec) {
        Objects.requireNonNull(res, "res must not be null");
        Objects.requireNonNull(vec, "vec must not be null");
        int n = Math.min(res.length, vec.length);
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment rSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, res);
            MemorySegment vSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vec);
            mujoco_h.mju_addTo(rSeg, vSeg, n);
            MemorySegment.copy(rSeg, ValueLayout.JAVA_DOUBLE, 0, res, 0, n);
        }
    }

    public static void subFrom(double[] res, double[] vec) {
        Objects.requireNonNull(res, "res must not be null");
        Objects.requireNonNull(vec, "vec must not be null");
        int n = Math.min(res.length, vec.length);
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment rSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, res);
            MemorySegment vSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vec);
            mujoco_h.mju_subFrom(rSeg, vSeg, n);
            MemorySegment.copy(rSeg, ValueLayout.JAVA_DOUBLE, 0, res, 0, n);
        }
    }

    public static void addToScl(double[] res, double[] vec, double scl) {
        Objects.requireNonNull(res, "res must not be null");
        Objects.requireNonNull(vec, "vec must not be null");
        int n = Math.min(res.length, vec.length);
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment rSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, res);
            MemorySegment vSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vec);
            mujoco_h.mju_addToScl(rSeg, vSeg, scl, n);
            MemorySegment.copy(rSeg, ValueLayout.JAVA_DOUBLE, 0, res, 0, n);
        }
    }

    public static void addScl(double[] res, double[] vec1, double[] vec2, double scl) {
        Objects.requireNonNull(res, "res must not be null");
        Objects.requireNonNull(vec1, "vec1 must not be null");
        Objects.requireNonNull(vec2, "vec2 must not be null");
        int n = Math.min(res.length, Math.min(vec1.length, vec2.length));
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment rSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, n);
            MemorySegment v1Seg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vec1);
            MemorySegment v2Seg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vec2);
            mujoco_h.mju_addScl(rSeg, v1Seg, v2Seg, scl, n);
            MemorySegment.copy(rSeg, ValueLayout.JAVA_DOUBLE, 0, res, 0, n);
        }
    }

    public static double normalize(double[] res) {
        Objects.requireNonNull(res, "res must not be null");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment rSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, res);
            double norm = mujoco_h.mju_normalize(rSeg, res.length);
            MemorySegment.copy(rSeg, ValueLayout.JAVA_DOUBLE, 0, res, 0, res.length);
            return norm;
        }
    }

    public static double norm(double[] res) {
        Objects.requireNonNull(res, "res must not be null");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment rSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, res);
            return mujoco_h.mju_norm(rSeg, res.length);
        }
    }

    public static double dot(double[] vec1, double[] vec2) {
        Objects.requireNonNull(vec1, "vec1 must not be null");
        Objects.requireNonNull(vec2, "vec2 must not be null");
        int n = Math.min(vec1.length, vec2.length);
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment v1Seg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vec1);
            MemorySegment v2Seg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vec2);
            return mujoco_h.mju_dot(v1Seg, v2Seg, n);
        }
    }

    public static void mulMatVec(double[] res, double[] mat, double[] vec, int nr, int nc) {
        checkLen(res, nr, "res");
        checkLen(mat, nr * nc, "mat");
        checkLen(vec, nc, "vec");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment rSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, nr);
            MemorySegment mSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, mat);
            MemorySegment vSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vec);
            mujoco_h.mju_mulMatVec(rSeg, mSeg, vSeg, nr, nc);
            MemorySegment.copy(rSeg, ValueLayout.JAVA_DOUBLE, 0, res, 0, nr);
        }
    }

    public static void mulMatTVec(double[] res, double[] mat, double[] vec, int nr, int nc) {
        checkLen(res, nc, "res");
        checkLen(mat, nr * nc, "mat");
        checkLen(vec, nr, "vec");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment rSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, nc);
            MemorySegment mSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, mat);
            MemorySegment vSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vec);
            mujoco_h.mju_mulMatTVec(rSeg, mSeg, vSeg, nr, nc);
            MemorySegment.copy(rSeg, ValueLayout.JAVA_DOUBLE, 0, res, 0, nc);
        }
    }

    public static double mulVecMatVec(double[] vec1, double[] mat, double[] vec2, int n) {
        checkLen(vec1, n, "vec1");
        checkLen(mat, n * n, "mat");
        checkLen(vec2, n, "vec2");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment v1Seg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vec1);
            MemorySegment mSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, mat);
            MemorySegment v2Seg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vec2);
            return mujoco_h.mju_mulVecMatVec(v1Seg, mSeg, v2Seg, n);
        }
    }

    public static void transpose(double[] res, double[] mat, int nr, int nc) {
        checkLen(res, nr * nc, "res");
        checkLen(mat, nr * nc, "mat");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment rSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, (long) nr * nc);
            MemorySegment mSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, mat);
            mujoco_h.mju_transpose(rSeg, mSeg, nr, nc);
            MemorySegment.copy(rSeg, ValueLayout.JAVA_DOUBLE, 0, res, 0, nr * nc);
        }
    }

    public static void symmetrize(double[] res, double[] mat, int n) {
        checkLen(res, n * n, "res");
        checkLen(mat, n * n, "mat");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment rSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, (long) n * n);
            MemorySegment mSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, mat);
            mujoco_h.mju_symmetrize(rSeg, mSeg, n);
            MemorySegment.copy(rSeg, ValueLayout.JAVA_DOUBLE, 0, res, 0, n * n);
        }
    }

    public static void eye(double[] mat, int n) {
        checkLen(mat, n * n, "mat");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment mSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, (long) n * n);
            mujoco_h.mju_eye(mSeg, n);
            MemorySegment.copy(mSeg, ValueLayout.JAVA_DOUBLE, 0, mat, 0, n * n);
        }
    }

    public static void mulMatMat(double[] res, double[] mat1, double[] mat2, int r1, int c1, int c2) {
        checkLen(res, r1 * c2, "res");
        checkLen(mat1, r1 * c1, "mat1");
        checkLen(mat2, c1 * c2, "mat2");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment rSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, (long) r1 * c2);
            MemorySegment m1Seg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, mat1);
            MemorySegment m2Seg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, mat2);
            mujoco_h.mju_mulMatMat(rSeg, m1Seg, m2Seg, r1, c1, c2);
            MemorySegment.copy(rSeg, ValueLayout.JAVA_DOUBLE, 0, res, 0, r1 * c2);
        }
    }

    public static void mulMatMatT(double[] res, double[] mat1, double[] mat2, int r1, int c1, int r2) {
        checkLen(res, r1 * r2, "res");
        checkLen(mat1, r1 * c1, "mat1");
        checkLen(mat2, r2 * c1, "mat2");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment rSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, (long) r1 * r2);
            MemorySegment m1Seg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, mat1);
            MemorySegment m2Seg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, mat2);
            mujoco_h.mju_mulMatMatT(rSeg, m1Seg, m2Seg, r1, c1, r2);
            MemorySegment.copy(rSeg, ValueLayout.JAVA_DOUBLE, 0, res, 0, r1 * r2);
        }
    }

    public static void mulMatTMat(double[] res, double[] mat1, double[] mat2, int r1, int c1, int c2) {
        checkLen(res, c1 * c2, "res");
        checkLen(mat1, r1 * c1, "mat1");
        checkLen(mat2, r1 * c2, "mat2");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment rSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, (long) c1 * c2);
            MemorySegment m1Seg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, mat1);
            MemorySegment m2Seg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, mat2);
            mujoco_h.mju_mulMatTMat(rSeg, m1Seg, m2Seg, r1, c1, c2);
            MemorySegment.copy(rSeg, ValueLayout.JAVA_DOUBLE, 0, res, 0, c1 * c2);
        }
    }

    public static void sqrMatTD(double[] res, double[] mat, double[] diag, int nr, int nc) {
        checkLen(res, nc * nc, "res");
        checkLen(mat, nr * nc, "mat");
        if (diag != null) checkLen(diag, nr, "diag");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment rSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, (long) nc * nc);
            MemorySegment mSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, mat);
            MemorySegment dSeg = diag != null ? temp.allocateFrom(ValueLayout.JAVA_DOUBLE, diag) : MemorySegment.NULL;
            mujoco_h.mju_sqrMatTD(rSeg, mSeg, dSeg, nr, nc);
            MemorySegment.copy(rSeg, ValueLayout.JAVA_DOUBLE, 0, res, 0, nc * nc);
        }
    }

    // =========================================================================
    // Sparse, Band, Cholesky, Eigenvalue & BoxQP Solvers
    // =========================================================================

    public static int dense2sparse(double[] res, double[] mat, int nr, int nc, int[] rownnz, int[] rowadr, int[] colind, int nnz) {
        checkLen(res, nnz, "res");
        checkLen(mat, nr * nc, "mat");
        checkLen(rownnz, nr, "rownnz");
        checkLen(rowadr, nr, "rowadr");
        checkLen(colind, nnz, "colind");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment rSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, nnz);
            MemorySegment mSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, mat);
            MemorySegment nnzSeg = temp.allocate(ValueLayout.JAVA_INT, nr);
            MemorySegment adrSeg = temp.allocate(ValueLayout.JAVA_INT, nr);
            MemorySegment colSeg = temp.allocate(ValueLayout.JAVA_INT, nnz);
            int code = mujoco_h.mju_dense2sparse(rSeg, mSeg, nr, nc, nnzSeg, adrSeg, colSeg, nnz);
            MemorySegment.copy(rSeg, ValueLayout.JAVA_DOUBLE, 0, res, 0, nnz);
            MemorySegment.copy(nnzSeg, ValueLayout.JAVA_INT, 0, rownnz, 0, nr);
            MemorySegment.copy(adrSeg, ValueLayout.JAVA_INT, 0, rowadr, 0, nr);
            MemorySegment.copy(colSeg, ValueLayout.JAVA_INT, 0, colind, 0, nnz);
            return code;
        }
    }

    public static void sparse2dense(double[] res, double[] mat, int nr, int nc, int[] rownnz, int[] rowadr, int[] colind) {
        checkLen(res, nr * nc, "res");
        Objects.requireNonNull(mat, "mat must not be null");
        checkLen(rownnz, nr, "rownnz");
        checkLen(rowadr, nr, "rowadr");
        Objects.requireNonNull(colind, "colind must not be null");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment rSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, (long) nr * nc);
            MemorySegment mSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, mat);
            MemorySegment nnzSeg = temp.allocateFrom(ValueLayout.JAVA_INT, rownnz);
            MemorySegment adrSeg = temp.allocateFrom(ValueLayout.JAVA_INT, rowadr);
            MemorySegment colSeg = temp.allocateFrom(ValueLayout.JAVA_INT, colind);
            mujoco_h.mju_sparse2dense(rSeg, mSeg, nr, nc, nnzSeg, adrSeg, colSeg);
            MemorySegment.copy(rSeg, ValueLayout.JAVA_DOUBLE, 0, res, 0, nr * nc);
        }
    }

    public static int cholFactor(double[] mat, int n, double mindiag) {
        checkLen(mat, n * n, "mat");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment mSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, mat);
            int rank = mujoco_h.mju_cholFactor(mSeg, n, mindiag);
            MemorySegment.copy(mSeg, ValueLayout.JAVA_DOUBLE, 0, mat, 0, n * n);
            return rank;
        }
    }

    public static void cholSolve(double[] res, double[] mat, double[] vec, int n) {
        checkLen(res, n, "res");
        checkLen(mat, n * n, "mat");
        checkLen(vec, n, "vec");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment rSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, n);
            MemorySegment mSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, mat);
            MemorySegment vSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vec);
            mujoco_h.mju_cholSolve(rSeg, mSeg, vSeg, n);
            MemorySegment.copy(rSeg, ValueLayout.JAVA_DOUBLE, 0, res, 0, n);
        }
    }

    public static int cholUpdate(double[] mat, double[] x, int n, int flgPlus) {
        checkLen(mat, n * n, "mat");
        checkLen(x, n, "x");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment mSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, mat);
            MemorySegment xSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, x);
            int rank = mujoco_h.mju_cholUpdate(mSeg, xSeg, n, flgPlus);
            MemorySegment.copy(mSeg, ValueLayout.JAVA_DOUBLE, 0, mat, 0, n * n);
            MemorySegment.copy(xSeg, ValueLayout.JAVA_DOUBLE, 0, x, 0, n);
            return rank;
        }
    }

    public static double cholFactorBand(double[] mat, int ntotal, int nband, int ndense, double diagadd, double diagmul) {
        Objects.requireNonNull(mat, "mat must not be null");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment mSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, mat);
            double minDiag = mujoco_h.mju_cholFactorBand(mSeg, ntotal, nband, ndense, diagadd, diagmul);
            MemorySegment.copy(mSeg, ValueLayout.JAVA_DOUBLE, 0, mat, 0, mat.length);
            return minDiag;
        }
    }

    public static void cholSolveBand(double[] res, double[] mat, double[] vec, int ntotal, int nband, int ndense) {
        checkLen(res, ntotal, "res");
        Objects.requireNonNull(mat, "mat must not be null");
        checkLen(vec, ntotal, "vec");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment rSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, ntotal);
            MemorySegment mSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, mat);
            MemorySegment vSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vec);
            mujoco_h.mju_cholSolveBand(rSeg, mSeg, vSeg, ntotal, nband, ndense);
            MemorySegment.copy(rSeg, ValueLayout.JAVA_DOUBLE, 0, res, 0, ntotal);
        }
    }

    public static void band2Dense(double[] res, double[] mat, int ntotal, int nband, int ndense, boolean flgSym) {
        checkLen(res, ntotal * ntotal, "res");
        Objects.requireNonNull(mat, "mat must not be null");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment rSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, (long) ntotal * ntotal);
            MemorySegment mSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, mat);
            mujoco_h.mju_band2Dense(rSeg, mSeg, ntotal, nband, ndense, flgSym);
            MemorySegment.copy(rSeg, ValueLayout.JAVA_DOUBLE, 0, res, 0, ntotal * ntotal);
        }
    }

    public static void dense2Band(double[] res, double[] mat, int ntotal, int nband, int ndense) {
        Objects.requireNonNull(res, "res must not be null");
        checkLen(mat, ntotal * ntotal, "mat");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment rSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, res.length);
            MemorySegment mSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, mat);
            mujoco_h.mju_dense2Band(rSeg, mSeg, ntotal, nband, ndense);
            MemorySegment.copy(rSeg, ValueLayout.JAVA_DOUBLE, 0, res, 0, res.length);
        }
    }

    public static void bandMulMatVec(double[] res, double[] mat, double[] vec, int ntotal, int nband, int ndense, int nvec, boolean flgSym) {
        checkLen(res, ntotal * nvec, "res");
        Objects.requireNonNull(mat, "mat must not be null");
        checkLen(vec, ntotal * nvec, "vec");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment rSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, (long) ntotal * nvec);
            MemorySegment mSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, mat);
            MemorySegment vSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vec);
            mujoco_h.mju_bandMulMatVec(rSeg, mSeg, vSeg, ntotal, nband, ndense, nvec, flgSym);
            MemorySegment.copy(rSeg, ValueLayout.JAVA_DOUBLE, 0, res, 0, ntotal * nvec);
        }
    }

    public static int bandDiag(int i, int ntotal, int nband, int ndense) {
        return mujoco_h.mju_bandDiag(i, ntotal, nband, ndense);
    }

    public static int eig3(double[] eigval, double[] eigvec, double[] quat, double[] mat) {
        checkLen(eigval, 3, "eigval");
        checkLen(eigvec, 9, "eigvec");
        checkLen(quat, 4, "quat");
        checkLen(mat, 9, "mat");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment valSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, 3);
            MemorySegment vecSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, 9);
            MemorySegment qSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, 4);
            MemorySegment mSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, mat);
            int iter = mujoco_h.mju_eig3(valSeg, vecSeg, qSeg, mSeg);
            MemorySegment.copy(valSeg, ValueLayout.JAVA_DOUBLE, 0, eigval, 0, 3);
            MemorySegment.copy(vecSeg, ValueLayout.JAVA_DOUBLE, 0, eigvec, 0, 9);
            MemorySegment.copy(qSeg, ValueLayout.JAVA_DOUBLE, 0, quat, 0, 4);
            return iter;
        }
    }

    public static int boxQP(double[] res, double[] R, int[] index, double[] H, double[] g, int n, double[] lower, double[] upper) {
        checkLen(res, n, "res");
        checkLen(R, n * (n + 7), "R");
        if (index != null) checkLen(index, n, "index");
        checkLen(H, n * n, "H");
        checkLen(g, n, "g");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment resSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, res);
            MemorySegment rSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, R.length);
            MemorySegment idxSeg = index != null ? temp.allocate(ValueLayout.JAVA_INT, n) : MemorySegment.NULL;
            MemorySegment hSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, H);
            MemorySegment gSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, g);
            MemorySegment lSeg = lower != null ? temp.allocateFrom(ValueLayout.JAVA_DOUBLE, lower) : MemorySegment.NULL;
            MemorySegment uSeg = upper != null ? temp.allocateFrom(ValueLayout.JAVA_DOUBLE, upper) : MemorySegment.NULL;
            int rank = mujoco_h.mju_boxQP(resSeg, rSeg, idxSeg, hSeg, gSeg, n, lSeg, uSeg);
            MemorySegment.copy(resSeg, ValueLayout.JAVA_DOUBLE, 0, res, 0, n);
            MemorySegment.copy(rSeg, ValueLayout.JAVA_DOUBLE, 0, R, 0, R.length);
            if (index != null) {
                MemorySegment.copy(idxSeg, ValueLayout.JAVA_INT, 0, index, 0, n);
            }
            return rank;
        }
    }

    // =========================================================================
    // Muscle, Pyramid, Scalar & String Utilities
    // =========================================================================

    public static double muscleGain(double len, double vel, double[] lengthrange, double acc0, double[] prm) {
        checkLen(lengthrange, 2, "lengthrange");
        checkLen(prm, 9, "prm");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment lrSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, lengthrange);
            MemorySegment prmSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, prm);
            return mujoco_h.mju_muscleGain(len, vel, lrSeg, acc0, prmSeg);
        }
    }

    public static double muscleBias(double len, double[] lengthrange, double acc0, double[] prm) {
        checkLen(lengthrange, 2, "lengthrange");
        checkLen(prm, 9, "prm");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment lrSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, lengthrange);
            MemorySegment prmSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, prm);
            return mujoco_h.mju_muscleBias(len, lrSeg, acc0, prmSeg);
        }
    }

    public static double muscleDynamics(double ctrl, double act, double[] prm) {
        checkLen(prm, 3, "prm");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment prmSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, prm);
            return mujoco_h.mju_muscleDynamics(ctrl, act, prmSeg);
        }
    }

    public static void encodePyramid(double[] pyramid, double[] force, double[] mu, int dim) {
        int npyr = 2 * (dim - 1);
        checkLen(pyramid, npyr, "pyramid");
        checkLen(force, dim, "force");
        checkLen(mu, dim - 1, "mu");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment pyrSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, Math.max(1, npyr));
            MemorySegment frcSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, force);
            MemorySegment muSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, mu);
            mujoco_h.mju_encodePyramid(pyrSeg, frcSeg, muSeg, dim);
            if (npyr > 0) MemorySegment.copy(pyrSeg, ValueLayout.JAVA_DOUBLE, 0, pyramid, 0, npyr);
        }
    }

    public static void decodePyramid(double[] force, double[] pyramid, double[] mu, int dim) {
        int npyr = 2 * (dim - 1);
        checkLen(force, dim, "force");
        checkLen(pyramid, npyr, "pyramid");
        checkLen(mu, dim - 1, "mu");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment frcSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, dim);
            MemorySegment pyrSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, pyramid);
            MemorySegment muSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, mu);
            mujoco_h.mju_decodePyramid(frcSeg, pyrSeg, muSeg, dim);
            MemorySegment.copy(frcSeg, ValueLayout.JAVA_DOUBLE, 0, force, 0, dim);
        }
    }

    public static double springDamper(double pos0, double vel0, double Kp, double Kv, double dt) {
        return mujoco_h.mju_springDamper(pos0, vel0, Kp, Kv, dt);
    }

    public static double min(double a, double b) {
        return mujoco_h.mju_min(a, b);
    }

    public static double max(double a, double b) {
        return mujoco_h.mju_max(a, b);
    }

    public static double clip(double x, double min, double max) {
        return mujoco_h.mju_clip(x, min, max);
    }

    public static double sign(double x) {
        return mujoco_h.mju_sign(x);
    }

    public static int round(double x) {
        return mujoco_h.mju_round(x);
    }

    public static double sigmoid(double x) {
        return mujoco_h.mju_sigmoid(x);
    }

    public static String type2Str(MjObjType type) {
        Objects.requireNonNull(type, "type must not be null");
        return type2Str(type.code());
    }

    public static String type2Str(int typeCode) {
        MemorySegment ptr = mujoco_h.mju_type2Str(typeCode);
        return ptr.equals(MemorySegment.NULL) ? null : ptr.reinterpret(Long.MAX_VALUE).getString(0);
    }

    public static int str2Type(String str) {
        Objects.requireNonNull(str, "str must not be null");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment sSeg = temp.allocateFrom(str);
            return mujoco_h.mju_str2Type(sSeg);
        }
    }

    public static String writeNumBytes(long nbytes) {
        MemorySegment ptr = mujoco_h.mju_writeNumBytes(nbytes);
        return ptr.equals(MemorySegment.NULL) ? "" : ptr.reinterpret(Long.MAX_VALUE).getString(0);
    }

    public static String warningText(int warning, long info) {
        MemorySegment ptr = mujoco_h.mju_warningText(warning, info);
        return ptr.equals(MemorySegment.NULL) ? "" : ptr.reinterpret(Long.MAX_VALUE).getString(0);
    }

    public static boolean isBad(double x) {
        return mujoco_h.mju_isBad(x) != 0;
    }

    public static boolean isZero(double[] vec) {
        Objects.requireNonNull(vec, "vec must not be null");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment vSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vec);
            return mujoco_h.mju_isZero(vSeg, vec.length) != 0;
        }
    }

    public static double standardNormal(double[] num2Out) {
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment n2Seg = num2Out != null && num2Out.length >= 1
                    ? temp.allocate(ValueLayout.JAVA_DOUBLE, 1)
                    : MemorySegment.NULL;
            double val = mujoco_h.mju_standardNormal(n2Seg);
            if (!n2Seg.equals(MemorySegment.NULL)) {
                num2Out[0] = n2Seg.get(ValueLayout.JAVA_DOUBLE, 0);
            }
            return val;
        }
    }

    public static void f2n(double[] res, float[] vec) {
        Objects.requireNonNull(res, "res must not be null");
        Objects.requireNonNull(vec, "vec must not be null");
        int n = Math.min(res.length, vec.length);
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment rSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, n);
            MemorySegment vSeg = temp.allocateFrom(ValueLayout.JAVA_FLOAT, vec);
            mujoco_h.mju_f2n(rSeg, vSeg, n);
            MemorySegment.copy(rSeg, ValueLayout.JAVA_DOUBLE, 0, res, 0, n);
        }
    }

    public static void n2f(float[] res, double[] vec) {
        Objects.requireNonNull(res, "res must not be null");
        Objects.requireNonNull(vec, "vec must not be null");
        int n = Math.min(res.length, vec.length);
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment rSeg = temp.allocate(ValueLayout.JAVA_FLOAT, n);
            MemorySegment vSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vec);
            mujoco_h.mju_n2f(rSeg, vSeg, n);
            MemorySegment.copy(rSeg, ValueLayout.JAVA_FLOAT, 0, res, 0, n);
        }
    }

    public static void d2n(double[] res, double[] vec) {
        copy(res, vec);
    }

    public static void n2d(double[] res, double[] vec) {
        copy(res, vec);
    }

    public static void insertionSort(double[] list) {
        Objects.requireNonNull(list, "list must not be null");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment lSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, list);
            mujoco_h.mju_insertionSort(lSeg, list.length);
            MemorySegment.copy(lSeg, ValueLayout.JAVA_DOUBLE, 0, list, 0, list.length);
        }
    }

    public static void insertionSortInt(int[] list) {
        Objects.requireNonNull(list, "list must not be null");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment lSeg = temp.allocateFrom(ValueLayout.JAVA_INT, list);
            mujoco_h.mju_insertionSortInt(lSeg, list.length);
            MemorySegment.copy(lSeg, ValueLayout.JAVA_INT, 0, list, 0, list.length);
        }
    }

    public static double halton(int index, int base) {
        return mujoco_h.mju_Halton(index, base);
    }

    private static void checkLen(double[] arr, int minLen, String name) {
        if (arr == null || arr.length < minLen) {
            throw new IllegalArgumentException(name + " must be non-null with length >= " + minLen);
        }
    }

    private static void checkLen(int[] arr, int minLen, String name) {
        if (arr == null || arr.length < minLen) {
            throw new IllegalArgumentException(name + " must be non-null with length >= " + minLen);
        }
    }
}
