package org.mujoco4j;

import org.mujoco4j.internal.panama.mujoco_h;

public enum MjTimer {
    STEP(mujoco_h.mjTIMER_STEP()),
    FORWARD(mujoco_h.mjTIMER_FORWARD()),
    INVERSE(mujoco_h.mjTIMER_INVERSE()),
    POSITION(mujoco_h.mjTIMER_POSITION()),
    VELOCITY(mujoco_h.mjTIMER_VELOCITY()),
    ACTUATION(mujoco_h.mjTIMER_ACTUATION()),
    CONSTRAINT(mujoco_h.mjTIMER_CONSTRAINT()),
    ADVANCE(mujoco_h.mjTIMER_ADVANCE()),
    POS_KINEMATICS(mujoco_h.mjTIMER_POS_KINEMATICS()),
    POS_INERTIA(mujoco_h.mjTIMER_POS_INERTIA()),
    POS_COLLISION(mujoco_h.mjTIMER_POS_COLLISION()),
    POS_MAKE(mujoco_h.mjTIMER_POS_MAKE()),
    POS_PROJECT(mujoco_h.mjTIMER_POS_PROJECT()),
    COL_BROAD(mujoco_h.mjTIMER_COL_BROAD()),
    COL_NARROW(mujoco_h.mjTIMER_COL_NARROW());

    public static final int NTIMER = mujoco_h.mjNTIMER();

    private final int value;

    MjTimer(int value) {
        this.value = value;
    }

    public int value() {
        return value;
    }

    public int code() {
        return value;
    }

    public static MjTimer fromInt(int value) {
        for (MjTimer v : values()) {
            if (v.value == value) {
                return v;
            }
        }
        throw new IllegalArgumentException("Invalid MjTimer value: " + value);
    }
}
