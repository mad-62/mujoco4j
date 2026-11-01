package org.mujoco4j;

import org.mujoco4j.internal.panama.mujoco_h;

public enum MjCatBit {
    STATIC(mujoco_h.mjCAT_STATIC()),
    DYNAMIC(mujoco_h.mjCAT_DYNAMIC()),
    DECOR(mujoco_h.mjCAT_DECOR()),
    ALL(mujoco_h.mjCAT_ALL());

    private final int value;

    MjCatBit(int value) {
        this.value = value;
    }

    public int value() {
        return value;
    }
}
