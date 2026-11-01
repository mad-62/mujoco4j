package org.mujoco4j;

import org.mujoco4j.internal.panama.mujoco_h;

public enum MjStage {
    NONE(mujoco_h.mjSTAGE_NONE()),
    POS(mujoco_h.mjSTAGE_POS()),
    VEL(mujoco_h.mjSTAGE_VEL()),
    ACC(mujoco_h.mjSTAGE_ACC());

    private final int value;

    MjStage(int value) {
        this.value = value;
    }

    public int value() {
        return value;
    }

    public int code() {
        return value;
    }

    public static MjStage fromInt(int value) {
        for (MjStage v : values()) {
            if (v.value == value) {
                return v;
            }
        }
        throw new IllegalArgumentException("Invalid MjStage value: " + value);
    }
}
