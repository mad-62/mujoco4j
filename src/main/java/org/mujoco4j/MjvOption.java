package org.mujoco4j;

import org.mujoco4j.internal.panama.mjvOption_;
import org.mujoco4j.internal.panama.mujoco_h;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * Wrapper around {@code mjvOption}.
 */
public final class MjvOption {

    private final MemorySegment segment;

    public MjvOption() {
        MjNativeLoader.ensureLoaded();
        this.segment = Arena.ofAuto().allocate(mjvOption_.layout());
        mujoco_h.mjv_defaultOption(this.segment);
    }

    public MemorySegment segment() {
        return segment;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof MjvOption other)) return false;
        return segment.mismatch(other.segment) == -1L;
    }

    @Override
    public int hashCode() {
        throw new UnsupportedOperationException("Mutable MuJoCo structs do not support hashCode()");
    }
}
