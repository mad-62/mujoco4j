package org.mujoco4j;

import org.mujoco4j.internal.panama.mujoco_h;

public enum MjEnableBit {
    OVERRIDE(mujoco_h.mjENBL_OVERRIDE()),
    ENERGY(mujoco_h.mjENBL_ENERGY()),
    FWDINV(mujoco_h.mjENBL_FWDINV()),
    INVDISCRETE(mujoco_h.mjENBL_INVDISCRETE()),
    SLEEP(mujoco_h.mjENBL_SLEEP()),
    DIAGEXACT(mujoco_h.mjENBL_DIAGEXACT()),
    IPC(mujoco_h.mjENBL_IPC());

    public static final int NENABLE = mujoco_h.mjNENABLE();

    private final int value;

    MjEnableBit(int value) {
        this.value = value;
    }

    public int value() {
        return value;
    }

    public static MjEnableBit fromInt(int value) {
        for (MjEnableBit v : values()) {
            if (v.value == value) {
                return v;
            }
        }
        throw new IllegalArgumentException("Invalid MjEnableBit value: " + value);
    }
}
