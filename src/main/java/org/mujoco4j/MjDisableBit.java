package org.mujoco4j;

import org.mujoco4j.internal.panama.mujoco_h;

public enum MjDisableBit {
    CONSTRAINT(mujoco_h.mjDSBL_CONSTRAINT()),
    EQUALITY(mujoco_h.mjDSBL_EQUALITY()),
    FRICTIONLOSS(mujoco_h.mjDSBL_FRICTIONLOSS()),
    LIMIT(mujoco_h.mjDSBL_LIMIT()),
    CONTACT(mujoco_h.mjDSBL_CONTACT()),
    SPRING(mujoco_h.mjDSBL_SPRING()),
    DAMPER(mujoco_h.mjDSBL_DAMPER()),
    GRAVITY(mujoco_h.mjDSBL_GRAVITY()),
    CLAMPCTRL(mujoco_h.mjDSBL_CLAMPCTRL()),
    WARMSTART(mujoco_h.mjDSBL_WARMSTART()),
    FILTERPARENT(mujoco_h.mjDSBL_FILTERPARENT()),
    ACTUATION(mujoco_h.mjDSBL_ACTUATION()),
    REFSAFE(mujoco_h.mjDSBL_REFSAFE()),
    SENSOR(mujoco_h.mjDSBL_SENSOR()),
    MIDPHASE(mujoco_h.mjDSBL_MIDPHASE()),
    EULERDAMP(mujoco_h.mjDSBL_EULERDAMP()),
    AUTORESET(mujoco_h.mjDSBL_AUTORESET()),
    NATIVECCD(mujoco_h.mjDSBL_NATIVECCD()),
    ISLAND(mujoco_h.mjDSBL_ISLAND()),
    MULTICCD(mujoco_h.mjDSBL_MULTICCD());

    public static final int NDISABLE = mujoco_h.mjNDISABLE();

    private final int value;

    MjDisableBit(int value) {
        this.value = value;
    }

    public int value() {
        return value;
    }

    public static MjDisableBit fromInt(int value) {
        for (MjDisableBit v : values()) {
            if (v.value == value) {
                return v;
            }
        }
        throw new IllegalArgumentException("Invalid MjDisableBit value: " + value);
    }
}
