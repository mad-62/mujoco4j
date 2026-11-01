package org.mujoco4j;

import org.mujoco4j.internal.panama.mujoco_h;

public enum MjGeomType {
    PLANE(mujoco_h.mjGEOM_PLANE()),
    HFIELD(mujoco_h.mjGEOM_HFIELD()),
    SPHERE(mujoco_h.mjGEOM_SPHERE()),
    CAPSULE(mujoco_h.mjGEOM_CAPSULE()),
    ELLIPSOID(mujoco_h.mjGEOM_ELLIPSOID()),
    CYLINDER(mujoco_h.mjGEOM_CYLINDER()),
    BOX(mujoco_h.mjGEOM_BOX()),
    MESH(mujoco_h.mjGEOM_MESH()),
    SDF(mujoco_h.mjGEOM_SDF()),
    NGEOMTYPES(mujoco_h.mjNGEOMTYPES()),
    ARROW(mujoco_h.mjGEOM_ARROW()),
    ARROW1(mujoco_h.mjGEOM_ARROW1()),
    ARROW2(mujoco_h.mjGEOM_ARROW2()),
    LINE(mujoco_h.mjGEOM_LINE()),
    LINEBOX(mujoco_h.mjGEOM_LINEBOX()),
    FLEX(mujoco_h.mjGEOM_FLEX()),
    SKIN(mujoco_h.mjGEOM_SKIN()),
    LABEL(mujoco_h.mjGEOM_LABEL()),
    TRIANGLE(mujoco_h.mjGEOM_TRIANGLE()),
    NONE(mujoco_h.mjGEOM_NONE());

    private final int value;

    MjGeomType(int value) {
        this.value = value;
    }

    public int value() {
        return value;
    }

    public static MjGeomType fromInt(int value) {
        for (MjGeomType v : values()) {
            if (v.value == value) {
                return v;
            }
        }
        throw new IllegalArgumentException("Invalid MjGeomType value: " + value);
    }
}
