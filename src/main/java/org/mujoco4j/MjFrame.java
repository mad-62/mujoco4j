package org.mujoco4j;

import org.mujoco4j.internal.panama.mujoco_h;

public enum MjFrame {
    NONE(mujoco_h.mjFRAME_NONE()),
    BODY(mujoco_h.mjFRAME_BODY()),
    GEOM(mujoco_h.mjFRAME_GEOM()),
    SITE(mujoco_h.mjFRAME_SITE()),
    CAMERA(mujoco_h.mjFRAME_CAMERA()),
    LIGHT(mujoco_h.mjFRAME_LIGHT()),
    CONTACT(mujoco_h.mjFRAME_CONTACT()),
    WORLD(mujoco_h.mjFRAME_WORLD());

    public static final int NFRAME = mujoco_h.mjNFRAME();

    private final int value;

    MjFrame(int value) {
        this.value = value;
    }

    public int value() {
        return value;
    }

    public static MjFrame fromInt(int value) {
        for (MjFrame v : values()) {
            if (v.value == value) {
                return v;
            }
        }
        throw new IllegalArgumentException("Invalid MjFrame value: " + value);
    }
}
