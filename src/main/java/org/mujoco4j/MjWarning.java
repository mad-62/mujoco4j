package org.mujoco4j;

import org.mujoco4j.internal.panama.mujoco_h;

public enum MjWarning {
    INERTIA(mujoco_h.mjWARN_INERTIA()),
    CONTACTFULL(mujoco_h.mjWARN_CONTACTFULL()),
    CNSTRFULL(mujoco_h.mjWARN_CNSTRFULL()),
    BADQPOS(mujoco_h.mjWARN_BADQPOS()),
    BADQVEL(mujoco_h.mjWARN_BADQVEL()),
    BADQACC(mujoco_h.mjWARN_BADQACC()),
    BADCTRL(mujoco_h.mjWARN_BADCTRL());

    public static final int NWARNING = mujoco_h.mjNWARNING();

    private final int value;

    MjWarning(int value) {
        this.value = value;
    }

    public int value() {
        return value;
    }

    public int code() {
        return value;
    }

    public static MjWarning fromInt(int value) {
        for (MjWarning v : values()) {
            if (v.value == value) {
                return v;
            }
        }
        throw new IllegalArgumentException("Invalid MjWarning value: " + value);
    }
}
