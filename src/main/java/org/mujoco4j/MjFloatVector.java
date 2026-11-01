package org.mujoco4j;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;
import java.util.function.BooleanSupplier;

/**
 * Bounds-checked, lifecycle-checked zero-copy view over a native {@code float*} array.
 */
public final class MjFloatVector {

    private final MemorySegment segment;
    private final int length;
    private final BooleanSupplier livenessGuard;

    public MjFloatVector(MemorySegment segment, int length, BooleanSupplier livenessGuard) {
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

    public float get(int index) {
        checkIndex(index);
        return segment.getAtIndex(ValueLayout.JAVA_FLOAT, index);
    }

    public void set(int index, float value) {
        checkIndex(index);
        segment.setAtIndex(ValueLayout.JAVA_FLOAT, index, value);
    }

    public MjFloatVector slice(int startElement, int elementCount) {
        checkOpen();
        if (startElement < 0 || elementCount < 0 || startElement + elementCount > length) {
            throw new IndexOutOfBoundsException(
                    "Slice [" + startElement + ", " + (startElement + elementCount) + ") out of bounds for length " + length);
        }
        if (elementCount == 0) {
            return new MjFloatVector(MemorySegment.NULL, 0, livenessGuard);
        }
        MemorySegment sub = segment.asSlice((long) startElement * Float.BYTES, (long) elementCount * Float.BYTES);
        return new MjFloatVector(sub, elementCount, livenessGuard);
    }

    public float[] toFloatArray() {
        checkOpen();
        if (length == 0) {
            return new float[0];
        }
        return segment.asSlice(0, (long) length * Float.BYTES).toArray(ValueLayout.JAVA_FLOAT);
    }

    public void copyFrom(float... values) {
        checkOpen();
        if (values == null || values.length != length) {
            throw new IllegalArgumentException("Expected length " + length + " but got " + (values == null ? "null" : values.length));
        }
        if (length > 0) {
            MemorySegment.copy(values, 0, segment, ValueLayout.JAVA_FLOAT, 0, length);
        }
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof MjFloatVector other)) return false;
        checkOpen();
        other.checkOpen();
        return Arrays.equals(toFloatArray(), other.toFloatArray());
    }

    @Override
    public int hashCode() {
        checkOpen();
        return Arrays.hashCode(toFloatArray());
    }

    @Override
    public String toString() {
        checkOpen();
        return Arrays.toString(toFloatArray());
    }
}
