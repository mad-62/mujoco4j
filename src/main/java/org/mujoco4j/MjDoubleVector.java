package org.mujoco4j;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;
import java.util.function.BooleanSupplier;

/**
 * Bounds-checked, lifecycle-checked zero-copy view over a native {@code mjtNum*} ({@code double*}) array
 * or 2D row-major matrix of shape {@code (rows, cols)}.
 */
public final class MjDoubleVector {

    private final MemorySegment segment;
    private final int length;
    private final int rows;
    private final int cols;
    private final BooleanSupplier livenessGuard;

    public MjDoubleVector(MemorySegment segment, int length, BooleanSupplier livenessGuard) {
        this(segment, length, 1, livenessGuard);
    }

    public MjDoubleVector(MemorySegment segment, int rows, int cols, BooleanSupplier livenessGuard) {
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

    private void checkRowCol(int row, int col) {
        checkOpen();
        if (row < 0 || row >= rows) {
            throw new IndexOutOfBoundsException("Row " + row + " out of bounds for rows " + rows);
        }
        if (col < 0 || col >= cols) {
            throw new IndexOutOfBoundsException("Col " + col + " out of bounds for cols " + cols);
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

    public int[] shape() {
        checkOpen();
        return cols == 1 ? new int[]{length} : new int[]{rows, cols};
    }

    public double get(int index) {
        checkIndex(index);
        return segment.getAtIndex(ValueLayout.JAVA_DOUBLE, index);
    }

    public void set(int index, double value) {
        checkIndex(index);
        segment.setAtIndex(ValueLayout.JAVA_DOUBLE, index, value);
    }

    public double get(int row, int col) {
        checkRowCol(row, col);
        return segment.getAtIndex(ValueLayout.JAVA_DOUBLE, (long) row * cols + col);
    }

    public void set(int row, int col, double value) {
        checkRowCol(row, col);
        segment.setAtIndex(ValueLayout.JAVA_DOUBLE, (long) row * cols + col, value);
    }

    public void fill(double value) {
        checkOpen();
        for (int i = 0; i < length; i++) {
            segment.setAtIndex(ValueLayout.JAVA_DOUBLE, i, value);
        }
    }

    public MjDoubleVector slice(int startElement, int elementCount) {
        checkOpen();
        if (startElement < 0 || elementCount < 0 || startElement + elementCount > length) {
            throw new IndexOutOfBoundsException(
                    "Slice [" + startElement + ", " + (startElement + elementCount) + ") out of bounds for length " + length);
        }
        if (elementCount == 0) {
            return new MjDoubleVector(MemorySegment.NULL, 0, 1, livenessGuard);
        }
        MemorySegment sub = segment.asSlice((long) startElement * Double.BYTES, (long) elementCount * Double.BYTES);
        return new MjDoubleVector(sub, elementCount, 1, livenessGuard);
    }

    public MjDoubleVector sliceRows(int startRow, int rowCount) {
        checkOpen();
        if (startRow < 0 || rowCount < 0 || startRow + rowCount > rows) {
            throw new IndexOutOfBoundsException(
                    "Row slice [" + startRow + ", " + (startRow + rowCount) + ") out of bounds for rows " + rows);
        }
        if (rowCount == 0 || cols == 0) {
            return new MjDoubleVector(MemorySegment.NULL, rowCount, cols, livenessGuard);
        }
        MemorySegment sub = segment.asSlice((long) startRow * cols * Double.BYTES, (long) rowCount * cols * Double.BYTES);
        return new MjDoubleVector(sub, rowCount, cols, livenessGuard);
    }

    public MjDoubleVector rowView(int row) {
        checkOpen();
        if (row < 0 || row >= rows) {
            throw new IndexOutOfBoundsException("Row " + row + " out of bounds for rows " + rows);
        }
        if (cols == 0) {
            return new MjDoubleVector(MemorySegment.NULL, 0, 1, livenessGuard);
        }
        MemorySegment sub = segment.asSlice((long) row * cols * Double.BYTES, (long) cols * Double.BYTES);
        return new MjDoubleVector(sub, cols, 1, livenessGuard);
    }

    public double[] getRow(int row) {
        return rowView(row).toDoubleArray();
    }

    public void setRow(int row, double... values) {
        MjDoubleVector rv = rowView(row);
        if (values.length == 1 && cols > 1) {
            rv.fill(values[0]);
        } else {
            rv.copyFrom(values);
        }
    }

    public double[] toDoubleArray() {
        checkOpen();
        if (length == 0) {
            return new double[0];
        }
        return segment.asSlice(0, (long) length * Double.BYTES).toArray(ValueLayout.JAVA_DOUBLE);
    }

    public double[][] toDoubleMatrix() {
        checkOpen();
        double[][] mat = new double[rows][cols];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                mat[r][c] = get(r, c);
            }
        }
        return mat;
    }

    public void copyFrom(double... values) {
        checkOpen();
        if (values == null) {
            throw new IllegalArgumentException("Source array must not be null");
        }
        if (values.length != length) {
            throw new IllegalArgumentException("Expected length " + length + " but got " + values.length);
        }
        if (length > 0) {
            MemorySegment.copy(values, 0, segment, ValueLayout.JAVA_DOUBLE, 0, length);
        }
    }

    public void copyFrom(MjDoubleVector other) {
        checkOpen();
        other.checkOpen();
        if (other.length != this.length) {
            throw new IllegalArgumentException("Expected length " + length + " but got " + other.length);
        }
        if (length > 0) {
            MemorySegment.copy(other.segment, 0, this.segment, 0, (long) length * Double.BYTES);
        }
    }

    public MemorySegment segment() {
        checkOpen();
        return segment;
    }

    public boolean sharesMemoryWith(MjDoubleVector other) {
        checkOpen();
        other.checkOpen();
        if (this.length == 0 || other.length == 0) {
            return false;
        }
        long aStart = this.segment.address();
        long aEnd = aStart + (long) this.length * Double.BYTES;
        long bStart = other.segment.address();
        long bEnd = bStart + (long) other.length * Double.BYTES;
        return aStart < bEnd && bStart < aEnd;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (!(obj instanceof MjDoubleVector other)) {
            return false;
        }
        checkOpen();
        other.checkOpen();
        if (this.rows != other.rows || this.cols != other.cols) {
            return false;
        }
        return Arrays.equals(this.toDoubleArray(), other.toDoubleArray());
    }

    @Override
    public int hashCode() {
        checkOpen();
        return Arrays.hashCode(toDoubleArray());
    }

    @Override
    public String toString() {
        checkOpen();
        return Arrays.toString(toDoubleArray());
    }
}
