package org.mujoco4j;

import org.mujoco4j.internal.panama.mujoco_h;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Objects;

/**
 * High-level raycasting functions wrapping MuJoCo's {@code mj_ray}, {@code mj_multiRay},
 * {@code mj_rayHfield}, {@code mj_rayMesh}, and {@code mju_rayGeom}.
 */
public final class MjRay {

    static {
        MjNativeLoader.ensureLoaded();
    }

    private MjRay() {}

    /**
     * Result of a single raycast query.
     *
     * @param distance Signed distance along ray vector {@code vec} to intersection, or {@code -1.0} if no hit
     * @param geomId   Index of intersecting geom, or {@code -1} if no hit
     * @param normal   3D surface normal at intersection (or {@code [0, 0, 0]} if no hit or not requested)
     */
    public record RayHit(double distance, int geomId, double[] normal) {}

    /**
     * Casts a single ray from {@code pnt} in direction {@code vec} (`mj_ray`).
     *
     * @param m           Model
     * @param d           Data (must have up-to-date kinematics, e.g. after {@link MjSim#forward})
     * @param pnt         Ray origin {@code [x, y, z]} (length >= 3)
     * @param vec         Ray direction {@code [dx, dy, dz]} (length >= 3)
     * @param geomgroup   Optional 6-byte geom group inclusion mask (may be {@code null} to include all groups)
     * @param flgStatic   Whether to include static geoms ({@code true} = include static geoms)
     * @param bodyexclude Body ID to exclude from intersection tests ({@code -1} to exclude none)
     * @param geomidOut   Optional int[1] output array to receive hit geom ID (may be {@code null})
     * @param normalOut   Optional double[3] output array to receive surface normal (may be {@code null})
     * @return Distance along {@code vec} to nearest intersection, or {@code -1.0} if no intersection
     */
    public static double ray(
            MjModel m, MjData d,
            double[] pnt, double[] vec,
            byte[] geomgroup, boolean flgStatic, int bodyexclude,
            int[] geomidOut, double[] normalOut) {
        Objects.requireNonNull(m, "m must not be null");
        Objects.requireNonNull(d, "d must not be null");
        m.checkOpen();
        d.checkOpen();
        if (pnt == null || pnt.length < 3 || vec == null || vec.length < 3) {
            throw new IllegalArgumentException("pnt and vec must have length >= 3");
        }
        if (geomgroup != null && geomgroup.length < MjConstants.mjNGROUP) {
            throw new IllegalArgumentException("geomgroup must have length >= " + MjConstants.mjNGROUP);
        }
        if (normalOut != null && normalOut.length < 3) {
            throw new IllegalArgumentException("normalOut must have length >= 3");
        }

        try (Arena temp = Arena.ofConfined()) {
            MemorySegment pntSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, pnt);
            MemorySegment vecSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vec);
            MemorySegment grpSeg = geomgroup != null
                    ? temp.allocateFrom(ValueLayout.JAVA_BYTE, geomgroup)
                    : MemorySegment.NULL;
            MemorySegment gidSeg = temp.allocate(ValueLayout.JAVA_INT, 1);
            gidSeg.set(ValueLayout.JAVA_INT, 0, -1);
            MemorySegment normSeg = normalOut != null
                    ? temp.allocate(ValueLayout.JAVA_DOUBLE, 3)
                    : MemorySegment.NULL;

            double dist = mujoco_h.mj_ray(
                    m.segment(), d.segment(),
                    pntSeg, vecSeg, grpSeg,
                    flgStatic, bodyexclude,
                    gidSeg, normSeg);

            if (geomidOut != null && geomidOut.length >= 1) {
                geomidOut[0] = gidSeg.get(ValueLayout.JAVA_INT, 0);
            }
            if (normalOut != null) {
                MemorySegment.copy(normSeg, ValueLayout.JAVA_DOUBLE, 0, normalOut, 0, 3);
            }
            return dist;
        }
    }

    /**
     * Convenience overload of {@link #ray} that returns a {@link RayHit} record with distance, geomId, and surface normal.
     */
    public static RayHit rayHit(
            MjModel m, MjData d,
            double[] pnt, double[] vec,
            byte[] geomgroup, boolean flgStatic, int bodyexclude) {
        int[] geomId = new int[]{-1};
        double[] normal = new double[3];
        double dist = ray(m, d, pnt, vec, geomgroup, flgStatic, bodyexclude, geomId, normal);
        return new RayHit(dist, geomId[0], normal);
    }

    /**
     * Casts {@code nray} rays originating from a single point {@code pnt} along directions {@code vec} (`mj_multiRay`).
     *
     * @param m           Model
     * @param d           Data
     * @param pnt         Common ray origin {@code [x, y, z]} (length >= 3)
     * @param vec         Ray directions {@code 3 * nray} (length >= {@code 3 * nray})
     * @param geomgroup   Optional 6-byte geom group mask (may be {@code null})
     * @param flgStatic   Whether to include static geoms
     * @param bodyexclude Body ID to exclude ({@code -1} for none)
     * @param geomidOut   Output array of length >= {@code nray} for hit geom IDs
     * @param distOut     Output array of length >= {@code nray} for hit distances
     * @param normalOut   Optional output array of length >= {@code 3 * nray} for surface normals (may be {@code null})
     * @param nray        Number of rays
     * @param cutoff      Maximum ray distance cutoff (e.g. {@link MjConstants#mjMAXVAL})
     */
    public static void multiRay(
            MjModel m, MjData d,
            double[] pnt, double[] vec,
            byte[] geomgroup, boolean flgStatic, int bodyexclude,
            int[] geomidOut, double[] distOut, double[] normalOut,
            int nray, double cutoff) {
        Objects.requireNonNull(m, "m must not be null");
        Objects.requireNonNull(d, "d must not be null");
        m.checkOpen();
        d.checkOpen();
        if (nray <= 0) {
            throw new IllegalArgumentException("nray must be > 0");
        }
        if (pnt == null || pnt.length < 3 || vec == null || vec.length < 3 * nray) {
            throw new IllegalArgumentException("pnt must have length >= 3 and vec >= 3*nray");
        }
        if (geomidOut == null || geomidOut.length < nray || distOut == null || distOut.length < nray) {
            throw new IllegalArgumentException("geomidOut and distOut must have length >= nray");
        }
        if (normalOut != null && normalOut.length < 3 * nray) {
            throw new IllegalArgumentException("normalOut must have length >= 3*nray");
        }

        try (Arena temp = Arena.ofConfined()) {
            MemorySegment pntSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, pnt);
            MemorySegment vecSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vec);
            MemorySegment grpSeg = geomgroup != null
                    ? temp.allocateFrom(ValueLayout.JAVA_BYTE, geomgroup)
                    : MemorySegment.NULL;
            MemorySegment gidSeg = temp.allocate(ValueLayout.JAVA_INT, nray);
            MemorySegment dstSeg = temp.allocate(ValueLayout.JAVA_DOUBLE, nray);
            MemorySegment nrmSeg = normalOut != null
                    ? temp.allocate(ValueLayout.JAVA_DOUBLE, 3L * nray)
                    : MemorySegment.NULL;

            mujoco_h.mj_multiRay(
                    m.segment(), d.segment(),
                    pntSeg, vecSeg, grpSeg,
                    flgStatic, bodyexclude,
                    gidSeg, dstSeg, nrmSeg,
                    nray, cutoff);

            MemorySegment.copy(gidSeg, ValueLayout.JAVA_INT, 0, geomidOut, 0, nray);
            MemorySegment.copy(dstSeg, ValueLayout.JAVA_DOUBLE, 0, distOut, 0, nray);
            if (normalOut != null) {
                MemorySegment.copy(nrmSeg, ValueLayout.JAVA_DOUBLE, 0, normalOut, 0, 3 * nray);
            }
        }
    }

    /**
     * Intersects a ray with a height field geom (`mj_rayHfield`).
     */
    public static double rayHfield(MjModel m, MjData d, int geomid, double[] pnt, double[] vec, double[] normalOut) {
        Objects.requireNonNull(m, "m must not be null");
        Objects.requireNonNull(d, "d must not be null");
        m.checkOpen();
        d.checkOpen();
        if (pnt == null || pnt.length < 3 || vec == null || vec.length < 3) {
            throw new IllegalArgumentException("pnt and vec must have length >= 3");
        }
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment pntSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, pnt);
            MemorySegment vecSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vec);
            MemorySegment nrmSeg = normalOut != null && normalOut.length >= 3
                    ? temp.allocate(ValueLayout.JAVA_DOUBLE, 3)
                    : MemorySegment.NULL;
            double dist = mujoco_h.mj_rayHfield(m.segment(), d.segment(), geomid, pntSeg, vecSeg, nrmSeg);
            if (!nrmSeg.equals(MemorySegment.NULL)) {
                MemorySegment.copy(nrmSeg, ValueLayout.JAVA_DOUBLE, 0, normalOut, 0, 3);
            }
            return dist;
        }
    }

    /**
     * Intersects a ray with a triangle mesh geom (`mj_rayMesh`).
     */
    public static double rayMesh(MjModel m, MjData d, int geomid, double[] pnt, double[] vec, double[] normalOut) {
        Objects.requireNonNull(m, "m must not be null");
        Objects.requireNonNull(d, "d must not be null");
        m.checkOpen();
        d.checkOpen();
        if (pnt == null || pnt.length < 3 || vec == null || vec.length < 3) {
            throw new IllegalArgumentException("pnt and vec must have length >= 3");
        }
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment pntSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, pnt);
            MemorySegment vecSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vec);
            MemorySegment nrmSeg = normalOut != null && normalOut.length >= 3
                    ? temp.allocate(ValueLayout.JAVA_DOUBLE, 3)
                    : MemorySegment.NULL;
            double dist = mujoco_h.mj_rayMesh(m.segment(), d.segment(), geomid, pntSeg, vecSeg, nrmSeg);
            if (!nrmSeg.equals(MemorySegment.NULL)) {
                MemorySegment.copy(nrmSeg, ValueLayout.JAVA_DOUBLE, 0, normalOut, 0, 3);
            }
            return dist;
        }
    }

    /**
     * Intersects a ray with a primitive analytical geom (`mju_rayGeom`).
     */
    public static double rayGeom(
            double[] pos, double[] mat, double[] size,
            double[] pnt, double[] vec, int geomtype,
            double[] normalOut) {
        if (pos == null || pos.length < 3 || mat == null || mat.length < 9 || size == null || size.length < 3
                || pnt == null || pnt.length < 3 || vec == null || vec.length < 3) {
            throw new IllegalArgumentException("Invalid array dimensions for rayGeom");
        }
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment posSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, pos);
            MemorySegment matSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, mat);
            MemorySegment szSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, size);
            MemorySegment pntSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, pnt);
            MemorySegment vecSeg = temp.allocateFrom(ValueLayout.JAVA_DOUBLE, vec);
            MemorySegment nrmSeg = normalOut != null && normalOut.length >= 3
                    ? temp.allocate(ValueLayout.JAVA_DOUBLE, 3)
                    : MemorySegment.NULL;
            double dist = mujoco_h.mju_rayGeom(posSeg, matSeg, szSeg, pntSeg, vecSeg, geomtype, nrmSeg);
            if (!nrmSeg.equals(MemorySegment.NULL)) {
                MemorySegment.copy(nrmSeg, ValueLayout.JAVA_DOUBLE, 0, normalOut, 0, 3);
            }
            return dist;
        }
    }
}
