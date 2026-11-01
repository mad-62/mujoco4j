package org.mujoco4j;

import org.mujoco4j.internal.panama.mjvFigure_;
import org.mujoco4j.internal.panama.mujoco_h;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * Wrapper around {@code mjvFigure}.
 */
public final class MjvFigure {

    private final MemorySegment segment;

    public MjvFigure() {
        MjNativeLoader.ensureLoaded();
        this.segment = Arena.ofAuto().allocate(mjvFigure_.layout());
        mujoco_h.mjv_defaultFigure(this.segment);
    }

    public MemorySegment segment() {
        return segment;
    }
}
