package org.mujoco4j;

import org.mujoco4j.internal.panama.mjvGLCamera_;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * Wrapper around {@code mjvGLCamera}.
 */
public final class MjvGLCamera {

    private final MemorySegment segment;

    public MjvGLCamera() {
        MjNativeLoader.ensureLoaded();
        this.segment = Arena.ofAuto().allocate(mjvGLCamera_.layout());
    }

    public MemorySegment segment() {
        return segment;
    }
}
