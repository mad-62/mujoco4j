package org.mujoco4j;

import org.mujoco4j.internal.panama.mujoco_h;

public enum MjState {
    TIME(mujoco_h.mjSTATE_TIME()),
    QPOS(mujoco_h.mjSTATE_QPOS()),
    QVEL(mujoco_h.mjSTATE_QVEL()),
    ACT(mujoco_h.mjSTATE_ACT()),
    HISTORY(mujoco_h.mjSTATE_HISTORY()),
    WARMSTART(mujoco_h.mjSTATE_WARMSTART()),
    CTRL(mujoco_h.mjSTATE_CTRL()),
    QFRC_APPLIED(mujoco_h.mjSTATE_QFRC_APPLIED()),
    XFRC_APPLIED(mujoco_h.mjSTATE_XFRC_APPLIED()),
    EQ_ACTIVE(mujoco_h.mjSTATE_EQ_ACTIVE()),
    MOCAP_POS(mujoco_h.mjSTATE_MOCAP_POS()),
    MOCAP_QUAT(mujoco_h.mjSTATE_MOCAP_QUAT()),
    USERDATA(mujoco_h.mjSTATE_USERDATA()),
    PLUGIN(mujoco_h.mjSTATE_PLUGIN()),
    PHYSICS(mujoco_h.mjSTATE_PHYSICS()),
    FULLPHYSICS(mujoco_h.mjSTATE_FULLPHYSICS()),
    USER(mujoco_h.mjSTATE_USER()),
    INTEGRATION(mujoco_h.mjSTATE_INTEGRATION());

    public static final int NSTATE = mujoco_h.mjNSTATE();

    private final int value;

    MjState(int value) {
        this.value = value;
    }

    public int value() {
        return value;
    }

    public int mask() {
        return value;
    }

    public int code() {
        return value;
    }
}
