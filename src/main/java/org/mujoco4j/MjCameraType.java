package org.mujoco4j;

import org.mujoco4j.internal.panama.mujoco_h;

public enum MjCameraType {
    FREE(mujoco_h.mjCAMERA_FREE()),
    TRACKING(mujoco_h.mjCAMERA_TRACKING()),
    FIXED(mujoco_h.mjCAMERA_FIXED()),
    USER(mujoco_h.mjCAMERA_USER());

    private final int value;

    MjCameraType(int value) {
        this.value = value;
    }

    public int value() {
        return value;
    }

    public static MjCameraType fromInt(int value) {
        for (MjCameraType v : values()) {
            if (v.value == value) {
                return v;
            }
        }
        throw new IllegalArgumentException("Invalid MjCameraType value: " + value);
    }
}
