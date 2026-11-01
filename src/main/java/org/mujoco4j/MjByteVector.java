package org.mujoco4j;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.function.BooleanSupplier;

/**
 * Bounds-checked, lifecycle-checked zero-copy view over a native {@code uint8_t*} / {@code char*} array.
 */
public final class MjByteVector {

    private final MemorySegment segment;
    private final int length;
    private final BooleanSupplier livenessGuard;

    public MjByteVector(MemorySegment segment, int length, BooleanSupplier livenessGuard) {
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

    public byte get(int index) {
        checkIndex(index);
        return segment.get(ValueLayout.JAVA_BYTE, index);
    }

    public int getUnsigned(int index) {
        return Byte.toUnsignedInt(get(index));
    }

    public void set(int index, byte value) {
        checkIndex(index);
        segment.set(ValueLayout.JAVA_BYTE, index, value);
    }

    public int indexOf(byte target, int fromIndex) {
        checkOpen();
        for (int i = Math.max(0, fromIndex); i < length; i++) {
            if (segment.get(ValueLayout.JAVA_BYTE, i) == target) {
                return i;
            }
        }
        return -1;
    }

    public String getNullTerminatedString(int startOffset) {
        checkOpen();
        if (startOffset < 0 || startOffset >= length) {
            throw new IndexOutOfBoundsException("Start offset " + startOffset + " out of bounds for length " + length);
        }
        int end = indexOf((byte) 0, startOffset);
        if (end < 0) {
            end = length;
        }
        byte[] bytes = segment.asSlice(startOffset, end - startOffset).toArray(ValueLayout.JAVA_BYTE);
        return new String(bytes, StandardCharsets.UTF_8);
    }

    public MjByteVector slice(int startOffset, int byteCount) {
        checkOpen();
        if (startOffset < 0 || byteCount < 0 || startOffset + byteCount > length) {
            throw new IndexOutOfBoundsException("Slice out of bounds");
        }
        if (byteCount == 0) {
            return new MjByteVector(MemorySegment.NULL, 0, livenessGuard);
        }
        return new MjByteVector(segment.asSlice(startOffset, byteCount), byteCount, livenessGuard);
    }

    public byte[] toByteArray() {
        checkOpen();
        if (length == 0) {
            return new byte[0];
        }
        return segment.asSlice(0, length).toArray(ValueLayout.JAVA_BYTE);
    }

    public boolean sharesMemoryWith(MjByteVector other) {
        checkOpen();
        other.checkOpen();
        if (this.length == 0 || other.length == 0) {
            return false;
        }
        long aStart = this.segment.address();
        long aEnd = aStart + this.length;
        long bStart = other.segment.address();
        long bEnd = bStart + other.length;
        return aStart < bEnd && bStart < aEnd;
    }

    public MemorySegment segment() {
        checkOpen();
        return segment;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof MjByteVector other)) return false;
        checkOpen();
        other.checkOpen();
        return Arrays.equals(toByteArray(), other.toByteArray());
    }

    @Override
    public int hashCode() {
        checkOpen();
        return Arrays.hashCode(toByteArray());
    }
}
