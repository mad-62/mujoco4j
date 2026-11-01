package org.mujoco4j;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;
import java.util.function.BooleanSupplier;

/**
 * Bounds-checked, lifecycle-checked zero-copy view over a native {@code int64_t*} ({@code mjtSize*}) array.
 */
public final class MjLongVector {

    private final MemorySegment segment;
    private final int length;
    private final BooleanSupplier livenessGuard;

    public MjLongVector(MemorySegment segment, int length, BooleanSupplier livenessGuard) {
        this.segment = segment;
        this.length = length;
        this.livenessGuard = livenessGuard != null ? livenessGuard : () -> true;
    }

    public void checkOpen() {
        if (!livenessGuard.getAsBoolean() || (segment != null && !MemorySegment.NULL.equals(segment) && !segment.scope().isAlive())) {
            throw new IllegalStateException("Underlying native MuJoCo memory has already been closed.");
        }
    }

    private void checkIndex(int index) {
        checkOpen();
        if (index < 0 || index >= length) {
            throw new IndexOutOfBoundsException("Index " + index + " out of bounds for length " + length);
        }
    }

    public int size() {
        checkOpen();
        return length;
    }

    public long get(int index) {
        checkIndex(index);
        return segment.getAtIndex(ValueLayout.JAVA_LONG, index);
    }

    public void set(int index, long value) {
        checkIndex(index);
        segment.setAtIndex(ValueLayout.JAVA_LONG, index, value);
    }

    public MjLongVector slice(int startElement, int elementCount) {
        checkOpen();
        if (startElement < 0 || elementCount < 0 || startElement + elementCount > length) {
            throw new IndexOutOfBoundsException(
                    "Slice [" + startElement + ", " + (startElement + elementCount) + ") out of bounds for length " + length);
        }
        if (elementCount == 0) {
            return new MjLongVector(MemorySegment.NULL, 0, livenessGuard);
        }
        MemorySegment sub = segment.asSlice((long) startElement * Long.BYTES, (long) elementCount * Long.BYTES);
        return new MjLongVector(sub, elementCount, livenessGuard);
    }

    public long[] toLongArray() {
        checkOpen();
        if (length == 0) {
            return new long[0];
        }
        return segment.asSlice(0, (long) length * Long.BYTES).toArray(ValueLayout.JAVA_LONG);
    }

    public boolean sharesMemoryWith(MjLongVector other) {
        checkOpen();
        other.checkOpen();
        if (this.length == 0 || other.length == 0) {
            return false;
        }
        long aStart = this.segment.address();
        long aEnd = aStart + (long) this.length * Long.BYTES;
        long bStart = other.segment.address();
        long bEnd = bStart + (long) other.length * Long.BYTES;
        return aStart < bEnd && bStart < aEnd;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof MjLongVector other)) return false;
        checkOpen();
        other.checkOpen();
        return Arrays.equals(toLongArray(), other.toLongArray());
    }

    @Override
    public int hashCode() {
        checkOpen();
        return Arrays.hashCode(toLongArray());
    }

    @Override
    public String toString() {
        checkOpen();
        return Arrays.toString(toLongArray());
    }
}
