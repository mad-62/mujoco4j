package org.mujoco4j;

import org.mujoco4j.internal.panama.mujoco_h;

public enum MjVisFlag {
    CONVEXHULL(mujoco_h.mjVIS_CONVEXHULL()),
    TEXTURE(mujoco_h.mjVIS_TEXTURE()),
    JOINT(mujoco_h.mjVIS_JOINT()),
    CAMERA(mujoco_h.mjVIS_CAMERA()),
    ACTUATOR(mujoco_h.mjVIS_ACTUATOR()),
    ACTIVATION(mujoco_h.mjVIS_ACTIVATION()),
    LIGHT(mujoco_h.mjVIS_LIGHT()),
    TENDON(mujoco_h.mjVIS_TENDON()),
    RANGEFINDER(mujoco_h.mjVIS_RANGEFINDER()),
    CONSTRAINT(mujoco_h.mjVIS_CONSTRAINT()),
    INERTIA(mujoco_h.mjVIS_INERTIA()),
    SCLINERTIA(mujoco_h.mjVIS_SCLINERTIA()),
    PERTFORCE(mujoco_h.mjVIS_PERTFORCE()),
    PERTOBJ(mujoco_h.mjVIS_PERTOBJ()),
    CONTACTPOINT(mujoco_h.mjVIS_CONTACTPOINT()),
    ISLAND(mujoco_h.mjVIS_ISLAND()),
    CONTACTFORCE(mujoco_h.mjVIS_CONTACTFORCE()),
    CONTACTSPLIT(mujoco_h.mjVIS_CONTACTSPLIT()),
    TRANSPARENT(mujoco_h.mjVIS_TRANSPARENT()),
    AUTOCONNECT(mujoco_h.mjVIS_AUTOCONNECT()),
    COM(mujoco_h.mjVIS_COM()),
    SELECT(mujoco_h.mjVIS_SELECT()),
    STATIC(mujoco_h.mjVIS_STATIC()),
    SKIN(mujoco_h.mjVIS_SKIN()),
    FLEXVERT(mujoco_h.mjVIS_FLEXVERT()),
    FLEXEDGE(mujoco_h.mjVIS_FLEXEDGE()),
    FLEXFACE(mujoco_h.mjVIS_FLEXFACE()),
    FLEXSKIN(mujoco_h.mjVIS_FLEXSKIN()),
    BODYBVH(mujoco_h.mjVIS_BODYBVH()),
    MESHBVH(mujoco_h.mjVIS_MESHBVH()),
    SDFITER(mujoco_h.mjVIS_SDFITER());

    public static final int NVISFLAG = mujoco_h.mjNVISFLAG();

    private final int value;

    MjVisFlag(int value) {
        this.value = value;
    }

    public int value() {
        return value;
    }
}
