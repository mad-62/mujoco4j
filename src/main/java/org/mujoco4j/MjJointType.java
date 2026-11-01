package org.mujoco4j;

import org.mujoco4j.internal.panama.mujoco_h;

public enum MjJointType {
    FREE(mujoco_h.mjJNT_FREE()),
    BALL(mujoco_h.mjJNT_BALL()),
    SLIDE(mujoco_h.mjJNT_SLIDE()),
    HINGE(mujoco_h.mjJNT_HINGE());

    private final int value;

    MjJointType(int value) {
        this.value = value;
    }

    public int value() {
        return value;
    }

    public int qposWidth() {
        return switch (this) {
            case FREE -> 7;
            case BALL -> 4;
            case SLIDE, HINGE -> 1;
        };
    }

    public int dofWidth() {
        return switch (this) {
            case FREE -> 6;
            case BALL -> 3;
            case SLIDE, HINGE -> 1;
        };
    }

    public static MjJointType fromInt(int value) {
        for (MjJointType v : values()) {
            if (v.value == value) {
                return v;
            }
        }
        throw new IllegalArgumentException("Invalid MjJointType value: " + value);
    }
}
