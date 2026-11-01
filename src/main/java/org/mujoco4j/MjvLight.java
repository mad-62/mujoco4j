package org.mujoco4j;

import org.mujoco4j.internal.panama.mjvLight_;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * Wrapper around {@code mjvLight}.
 */
public final class MjvLight {

    private final MemorySegment segment;

    public MjvLight() {
        MjNativeLoader.ensureLoaded();
        this.segment = Arena.ofAuto().allocate(mjvLight_.layout());
    }

    public MemorySegment segment() {
        return segment;
    }
}
