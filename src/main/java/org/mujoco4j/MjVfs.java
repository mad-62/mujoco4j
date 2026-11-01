package org.mujoco4j;

import org.mujoco4j.internal.panama.mjVFS_;
import org.mujoco4j.internal.panama.mujoco_h;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Safe {@link AutoCloseable} wrapper around MuJoCo's virtual file system ({@code mjVFS}).
 */
public final class MjVfs implements AutoCloseable {

    private final Arena arena;
    private final MemorySegment vfsSeg;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    public MjVfs() {
        MjNativeLoader.ensureLoaded();
        this.arena = Arena.ofShared();
        MemorySegment raw = arena.allocate(mjVFS_.layout());
        mujoco_h.mj_defaultVFS(raw);
        this.vfsSeg = raw.reinterpret(mjVFS_.sizeof(), arena, mujoco_h::mj_deleteVFS);
    }

    public void checkOpen() {
        if (closed.get() || !arena.scope().isAlive()) {
            throw new IllegalStateException("MjVfs has already been closed.");
        }
    }

    public void addBuffer(String filename, byte[] content) {
        checkOpen();
        if (filename == null || content == null) {
            throw new IllegalArgumentException("filename and content must not be null");
        }
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment nameStr = temp.allocateFrom(filename);
            MemorySegment bufSeg = temp.allocate(content.length);
            if (content.length > 0) {
                MemorySegment.copy(content, 0, bufSeg, ValueLayout.JAVA_BYTE, 0, content.length);
            }
            int rc = mujoco_h.mj_addBufferVFS(vfsSeg, nameStr, bufSeg, content.length);
            if (rc == 2) {
                throw new MjException("Repeated file name in VFS: " + filename);
            } else if (rc != 0) {
                throw new MjException("Failed to add buffer to VFS: " + filename + " (code " + rc + ")");
            }
        }
    }

    public boolean containsBuffer(String filename) {
        checkOpen();
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment nameStr = temp.allocateFrom(filename);
            return mujoco_h.mj_containsBufferVFS(vfsSeg, nameStr) == 1;
        }
    }

    public void deleteFile(String filename) {
        checkOpen();
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment nameStr = temp.allocateFrom(filename);
            int rc = mujoco_h.mj_deleteFileVFS(vfsSeg, nameStr);
            if (rc != 0) {
                throw new MjException("File not found in VFS: " + filename);
            }
        }
    }

    public MemorySegment segment() {
        checkOpen();
        return vfsSeg;
    }

    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            arena.close();
        }
    }
}
