package org.mujoco4j;

import org.mujoco4j.internal.panama.mjvPerturb_;
import org.mujoco4j.internal.panama.mujoco_h;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * Wrapper around {@code mjvPerturb}.
 */
public final class MjvPerturb {

    private final MemorySegment segment;

    public MjvPerturb() {
        MjNativeLoader.ensureLoaded();
        this.segment = Arena.ofAuto().allocate(mjvPerturb_.layout());
        mujoco_h.mjv_defaultPerturb(this.segment);
    }

    public int select() {
        return mjvPerturb_.select(segment);
    }

    public void setSelect(int select) {
        mjvPerturb_.select(segment, select);
    }

    public MemorySegment segment() {
        return segment;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof MjvPerturb other)) return false;
        return segment.mismatch(other.segment) == -1L;
    }

    @Override
    public int hashCode() {
        throw new UnsupportedOperationException("Mutable MuJoCo structs do not support hashCode()");
    }

    @Override
    public String toString() {
        return "<MjvPerturb select: " + select() + ">";
    }
}
