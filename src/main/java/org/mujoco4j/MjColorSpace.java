package org.mujoco4j;

import org.mujoco4j.internal.panama.mujoco_h;

public enum MjColorSpace {
    AUTO(mujoco_h.mjCOLORSPACE_AUTO()),
    LINEAR(mujoco_h.mjCOLORSPACE_LINEAR()),
    SRGB(mujoco_h.mjCOLORSPACE_SRGB());

    private final int value;

    MjColorSpace(int value) {
        this.value = value;
    }

    public int value() {
        return value;
    }

    public static MjColorSpace fromInt(int value) {
        for (MjColorSpace v : values()) {
            if (v.value == value) {
                return v;
            }
        }
        throw new IllegalArgumentException("Invalid MjColorSpace value: " + value);
    }
}
