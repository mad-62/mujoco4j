package org.mujoco4j;

import org.mujoco4j.internal.panama.mujoco_h;

public enum MjObjType {
    UNKNOWN(mujoco_h.mjOBJ_UNKNOWN()),
    BODY(mujoco_h.mjOBJ_BODY()),
    XBODY(mujoco_h.mjOBJ_XBODY()),
    JOINT(mujoco_h.mjOBJ_JOINT()),
    DOF(mujoco_h.mjOBJ_DOF()),
    GEOM(mujoco_h.mjOBJ_GEOM()),
    SITE(mujoco_h.mjOBJ_SITE()),
    CAMERA(mujoco_h.mjOBJ_CAMERA()),
    LIGHT(mujoco_h.mjOBJ_LIGHT()),
    FLEX(mujoco_h.mjOBJ_FLEX()),
    MESH(mujoco_h.mjOBJ_MESH()),
    SKIN(mujoco_h.mjOBJ_SKIN()),
    HFIELD(mujoco_h.mjOBJ_HFIELD()),
    TEXTURE(mujoco_h.mjOBJ_TEXTURE()),
    MATERIAL(mujoco_h.mjOBJ_MATERIAL()),
    PAIR(mujoco_h.mjOBJ_PAIR()),
    EXCLUDE(mujoco_h.mjOBJ_EXCLUDE()),
    EQUALITY(mujoco_h.mjOBJ_EQUALITY()),
    TENDON(mujoco_h.mjOBJ_TENDON()),
    ACTUATOR(mujoco_h.mjOBJ_ACTUATOR()),
    SENSOR(mujoco_h.mjOBJ_SENSOR()),
    NUMERIC(mujoco_h.mjOBJ_NUMERIC()),
    TEXT(mujoco_h.mjOBJ_TEXT()),
    TUPLE(mujoco_h.mjOBJ_TUPLE()),
    KEY(mujoco_h.mjOBJ_KEY()),
    PLUGIN(mujoco_h.mjOBJ_PLUGIN()),
    NOBJECT(mujoco_h.mjNOBJECT()),
    FRAME(mujoco_h.mjOBJ_FRAME()),
    DEFAULT(mujoco_h.mjOBJ_DEFAULT()),
    MODEL(mujoco_h.mjOBJ_MODEL());

    private final int value;

    MjObjType(int value) {
        this.value = value;
    }

    public int value() {
        return value;
    }

    public int code() {
        return value;
    }

    public static MjObjType fromInt(int value) {
        for (MjObjType v : values()) {
            if (v.value == value) {
                return v;
            }
        }
        throw new IllegalArgumentException("Invalid MjObjType value: " + value);
    }
}
