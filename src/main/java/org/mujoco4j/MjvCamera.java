package org.mujoco4j;

import org.mujoco4j.internal.panama.mjvCamera_;
import org.mujoco4j.internal.panama.mujoco_h;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * Wrapper around {@code mjvCamera}.
 */
public final class MjvCamera {

    private final MemorySegment segment;

    public MjvCamera() {
        MjNativeLoader.ensureLoaded();
        this.segment = Arena.ofAuto().allocate(mjvCamera_.layout());
        mujoco_h.mjv_defaultCamera(this.segment);
    }

    public int type() {
        return mjvCamera_.type(segment);
    }

    public void setType(int type) {
        mjvCamera_.type(segment, type);
    }

    public void setType(MjCameraType type) {
        setType(type.value());
    }

    public int fixedcamid() {
        return mjvCamera_.fixedcamid(segment);
    }

    public void setFixedcamid(int fixedcamid) {
        mjvCamera_.fixedcamid(segment, fixedcamid);
    }

    public int trackbodyid() {
        return mjvCamera_.trackbodyid(segment);
    }

    public void setTrackbodyid(int trackbodyid) {
        mjvCamera_.trackbodyid(segment, trackbodyid);
    }

    public MemorySegment segment() {
        return segment;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof MjvCamera other)) return false;
        return segment.mismatch(other.segment) == -1L;
    }

    @Override
    public int hashCode() {
        throw new UnsupportedOperationException("Mutable MuJoCo structs do not support hashCode()");
    }

    @Override
    public String toString() {
        return "<MjvCamera fixedcamid: " + fixedcamid() + ", type: " + type() + ">";
    }
}
