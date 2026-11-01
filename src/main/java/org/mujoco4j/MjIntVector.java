package org.mujoco4j;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;
import java.util.function.BooleanSupplier;

/**
 * Bounds-checked, lifecycle-checked zero-copy view over a native {@code int*} array
 * or 2D row-major matrix of shape {@code (rows, cols)}.
 */
public final class MjIntVector {

    private final MemorySegment segment;
    private final int length;
    private final int rows;
    private final int cols;
    private final BooleanSupplier livenessGuard;

    public MjIntVector(MemorySegment segment, int length, BooleanSupplier livenessGuard) {
        this(segment, length, 1, livenessGuard);
    }

    public MjIntVector(MemorySegment segment, int rows, int cols, BooleanSupplier livenessGuard) {
        this.segment = segment;
        this.rows = rows;
        this.cols = cols;
        this.length = Math.multiplyExact(rows, cols);
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

    public int rows() {
        checkOpen();
        return rows;
    }

    public int cols() {
        checkOpen();
        return cols;
    }

    public int get(int index) {
        checkIndex(index);
        return segment.getAtIndex(ValueLayout.JAVA_INT, index);
    }

    public void set(int index, int value) {
        checkIndex(index);
        segment.setAtIndex(ValueLayout.JAVA_INT, index, value);
    }

    public int get(int row, int col) {
        checkOpen();
        if (row < 0 || row >= rows || col < 0 || col >= cols) {
            throw new IndexOutOfBoundsException("Row/Col (" + row + ", " + col + ") out of bounds for (" + rows + ", " + cols + ")");
        }
        return segment.getAtIndex(ValueLayout.JAVA_INT, (long) row * cols + col);
    }

    public void set(int row, int col, int value) {
        checkOpen();
        if (row < 0 || row >= rows || col < 0 || col >= cols) {
            throw new IndexOutOfBoundsException("Row/Col (" + row + ", " + col + ") out of bounds for (" + rows + ", " + cols + ")");
        }
        segment.setAtIndex(ValueLayout.JAVA_INT, (long) row * cols + col, value);
    }

    public MjIntVector slice(int startElement, int elementCount) {
        checkOpen();
        if (startElement < 0 || elementCount < 0 || startElement + elementCount > length) {
            throw new IndexOutOfBoundsException(
                    "Slice [" + startElement + ", " + (startElement + elementCount) + ") out of bounds for length " + length);
        }
        if (elementCount == 0) {
            return new MjIntVector(MemorySegment.NULL, 0, 1, livenessGuard);
        }
        MemorySegment sub = segment.asSlice((long) startElement * Integer.BYTES, (long) elementCount * Integer.BYTES);
        return new MjIntVector(sub, elementCount, 1, livenessGuard);
    }

    public int[] toIntArray() {
        checkOpen();
        if (length == 0) {
            return new int[0];
        }
        return segment.asSlice(0, (long) length * Integer.BYTES).toArray(ValueLayout.JAVA_INT);
    }

    public void copyFrom(int... values) {
        checkOpen();
        if (values == null || values.length != length) {
            throw new IllegalArgumentException("Expected length " + length + " but got " + (values == null ? "null" : values.length));
        }
        if (length > 0) {
            MemorySegment.copy(values, 0, segment, ValueLayout.JAVA_INT, 0, length);
        }
    }

    public MemorySegment segment() {
        checkOpen();
        return segment;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof MjIntVector other)) return false;
        checkOpen();
        other.checkOpen();
        return this.rows == other.rows && this.cols == other.cols && Arrays.equals(toIntArray(), other.toIntArray());
    }

    @Override
    public int hashCode() {
        checkOpen();
        return Arrays.hashCode(toIntArray());
    }

    @Override
    public String toString() {
        checkOpen();
        return Arrays.toString(toIntArray());
    }
}
