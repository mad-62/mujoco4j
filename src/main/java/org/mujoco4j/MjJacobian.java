package org.mujoco4j;

import org.mujoco4j.internal.panama.mujoco_h;

public enum MjJacobian {
    DENSE(mujoco_h.mjJAC_DENSE()),
    SPARSE(mujoco_h.mjJAC_SPARSE()),
    AUTO(mujoco_h.mjJAC_AUTO());

    private final int value;

    MjJacobian(int value) {
        this.value = value;
    }

    public int value() {
        return value;
    }

    public static MjJacobian fromInt(int value) {
        for (MjJacobian v : values()) {
            if (v.value == value) {
                return v;
            }
        }
        throw new IllegalArgumentException("Invalid MjJacobian value: " + value);
    }
}
