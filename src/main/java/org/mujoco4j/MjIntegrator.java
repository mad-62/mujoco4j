package org.mujoco4j;

import org.mujoco4j.internal.panama.mujoco_h;

public enum MjIntegrator {
    EULER(mujoco_h.mjINT_EULER()),
    RK4(mujoco_h.mjINT_RK4()),
    IMPLICIT(mujoco_h.mjINT_IMPLICIT()),
    IMPLICITFAST(mujoco_h.mjINT_IMPLICITFAST());

    private final int value;

    MjIntegrator(int value) {
        this.value = value;
    }

    public int value() {
        return value;
    }

    public int code() {
        return value;
    }

    public static MjIntegrator fromInt(int value) {
        for (MjIntegrator v : values()) {
            if (v.value == value) {
                return v;
            }
        }
        throw new IllegalArgumentException("Invalid MjIntegrator value: " + value);
    }
}
