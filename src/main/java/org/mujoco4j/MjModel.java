package org.mujoco4j;

import org.mujoco4j.internal.panama.mjModel_;
import org.mujoco4j.internal.panama.mujoco_h;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * High-level, safe, zero-copy wrapper around MuJoCo's {@code mjModel} struct.
 * <p>
 * Manages native model memory via a shared {@link Arena} and provides bounds-checked,
 * lifetime-checked views over model parameters, geometry, joints, actuators, sensors,
 * textures, and named element accessors.
 */
public final class MjModel implements AutoCloseable {

    static {
        MjNativeLoader.ensureLoaded();
    }

    private final Arena arena;
    private final MemorySegment segment;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    // Cached sub-struct views
    private final MjOption optView;
    private final MjStatistic statView;
    private final MjVisual visView;

    private MjModel(MemorySegment rawModelPtr) {
        if (rawModelPtr == null || rawModelPtr.equals(MemorySegment.NULL)) {
            throw new MjException("Failed to allocate or compile mjModel (null pointer returned)");
        }
        this.arena = Arena.ofShared();
        this.segment = rawModelPtr.reinterpret(mjModel_.sizeof(), arena, mujoco_h::mj_deleteModel);
        this.optView = new MjOption(mjModel_.opt(this.segment), this::isOpen);
        this.statView = new MjStatistic(mjModel_.stat(this.segment), this::isOpen);
        this.visView = new MjVisual(mjModel_.vis(this.segment), this::isOpen);
        MjCallbacks.registerModel(this);
    }

    // =========================================================================
    // Factory Methods
    // =========================================================================

    /**
     * Parses and compiles an MJCF or URDF model from an XML string.
     */
    public static MjModel fromXmlString(String xml) {
        return fromXmlString(xml, null);
    }

    /**
     * Parses and compiles an MJCF or URDF model from an XML string with virtual file assets.
     */
    public static MjModel fromXmlString(String xml, Map<String, byte[]> assets) {
        Objects.requireNonNull(xml, "xml must not be null");
        if (assets != null && !assets.isEmpty()) {
            try (MjVfs vfs = new MjVfs()) {
                for (Map.Entry<String, byte[]> entry : assets.entrySet()) {
                    vfs.addBuffer(entry.getKey(), entry.getValue());
                }
                return fromXmlStringWithVfs(xml, vfs);
            }
        } else {
            return fromXmlStringWithVfs(xml, null);
        }
    }

    /**
     * Parses and compiles an MJCF or URDF model from an XML string using an explicit {@link MjVfs}.
     */
    public static MjModel fromXmlStringWithVfs(String xml, MjVfs vfs) {
        Objects.requireNonNull(xml, "xml must not be null");
        MemorySegment vfsSeg = vfs != null ? vfs.segment() : MemorySegment.NULL;
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment xmlSeg = temp.allocateFrom(xml);
            MemorySegment errBuf = temp.allocate(1024);
            MemorySegment spec = mujoco_h.mj_parseXMLString(xmlSeg, vfsSeg, errBuf, 1024);
            if (spec.equals(MemorySegment.NULL)) {
                String errorMsg = errBuf.getString(0);
                throw new MjException("Failed to parse XML string: " + errorMsg);
            }
            try {
                MemorySegment rawModel = mujoco_h.mj_compile(spec, vfsSeg);
                if (rawModel.equals(MemorySegment.NULL)) {
                    MemorySegment specErr = mujoco_h.mjs_getError(spec);
                    String errMsg = specErr.equals(MemorySegment.NULL)
                            ? "Unknown compilation error"
                            : specErr.reinterpret(Long.MAX_VALUE).getString(0);
                    throw new MjException("Failed to compile model from spec: " + errMsg);
                }
                return new MjModel(rawModel);
            } finally {
                mujoco_h.mj_deleteSpec(spec);
            }
        }
    }

    /**
     * Loads and compiles an MJCF or URDF model from a file path on disk.
     */
    public static MjModel fromXmlPath(Path path) {
        return fromXmlPath(path, null);
    }

    /**
     * Loads and compiles an MJCF or URDF model from a file path or VFS path.
     */
    public static MjModel fromXmlPath(Path path, MjVfs vfs) {
        Objects.requireNonNull(path, "path must not be null");
        return fromXmlFilename(path.toString(), vfs);
    }

    /**
     * Loads and compiles an MJCF or URDF model from a filename (on disk or in VFS).
     */
    public static MjModel fromXmlFilename(String filename, MjVfs vfs) {
        Objects.requireNonNull(filename, "filename must not be null");
        MemorySegment vfsSeg = vfs != null ? vfs.segment() : MemorySegment.NULL;
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment fileSeg = temp.allocateFrom(filename);
            MemorySegment errBuf = temp.allocate(1024);
            MemorySegment rawModel = mujoco_h.mj_loadXML(fileSeg, vfsSeg, errBuf, 1024);
            if (rawModel.equals(MemorySegment.NULL)) {
                String errorMsg = errBuf.getString(0);
                throw new MjException("Failed to load XML '" + filename + "': " + errorMsg);
            }
            return new MjModel(rawModel);
        }
    }

    /**
     * Loads a compiled binary MJB model from a file path on disk.
     */
    public static MjModel fromBinaryPath(Path path) {
        Objects.requireNonNull(path, "path must not be null");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment fileSeg = temp.allocateFrom(path.toString());
            MemorySegment rawModel = mujoco_h.mj_loadModel(fileSeg, MemorySegment.NULL);
            if (rawModel.equals(MemorySegment.NULL)) {
                throw new MjException("Failed to load binary model from '" + path + "'");
            }
            return new MjModel(rawModel);
        }
    }

    /**
     * Loads a compiled binary MJB model from an in-memory byte array.
     */
    public static MjModel fromBinaryBuffer(byte[] mjbBytes) {
        Objects.requireNonNull(mjbBytes, "mjbBytes must not be null");
        if (mjbBytes.length == 0) {
            throw new MjException("Binary model buffer is empty");
        }
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment bufSeg = temp.allocate(mjbBytes.length);
            MemorySegment.copy(mjbBytes, 0, bufSeg, ValueLayout.JAVA_BYTE, 0, mjbBytes.length);
            MemorySegment rawModel = mujoco_h.mj_loadModelBuffer(bufSeg, mjbBytes.length);
            if (rawModel.equals(MemorySegment.NULL)) {
                throw new MjException("Failed to load binary model from buffer (" + mjbBytes.length + " bytes)");
            }
            return new MjModel(rawModel);
        }
    }

    public static MjModel loadFromXmlString(String xml) {
        return fromXmlString(xml);
    }

    public static MjModel loadFromXmlString(String xml, MjVfs vfs) {
        return fromXmlStringWithVfs(xml, vfs);
    }

    public static MjModel loadFromXmlPath(Path path) {
        return fromXmlPath(path);
    }

    public static MjModel loadFromXmlPath(Path path, MjVfs vfs) {
        return fromXmlPath(path, vfs);
    }

    public static MjModel loadFromBinaryPath(Path path) {
        return fromBinaryPath(path);
    }

    public static MjModel loadFromBinaryPath(Path path, MjVfs vfs) {
        Objects.requireNonNull(path, "path must not be null");
        MemorySegment vfsSeg = vfs != null ? vfs.segment() : MemorySegment.NULL;
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment fileSeg = temp.allocateFrom(path.toString());
            MemorySegment rawModel = mujoco_h.mj_loadModel(fileSeg, vfsSeg);
            if (rawModel.equals(MemorySegment.NULL)) {
                throw new MjException("Failed to load binary model from '" + path + "'");
            }
            return new MjModel(rawModel);
        }
    }

    public static MjModel loadFromBinaryBuffer(byte[] mjbBytes) {
        return fromBinaryBuffer(mjbBytes);
    }

    // =========================================================================
    // Lifecycle & Model Operations
    // =========================================================================

    /**
     * Creates a new {@link MjData} instance associated with this model.
     */
    public MjData makeData() {
        checkOpen();
        return new MjData(this);
    }

    /**
     * Creates a deep copy of this model.
     */
    public MjModel copy() {
        checkOpen();
        MemorySegment copiedPtr = mujoco_h.mj_copyModel(MemorySegment.NULL, this.segment);
        return new MjModel(copiedPtr);
    }

    /**
     * Copies this model's contents into an existing destination model.
     */
    public void copyTo(MjModel dest) {
        checkOpen();
        Objects.requireNonNull(dest, "dest must not be null");
        dest.checkOpen();
        MemorySegment res = mujoco_h.mj_copyModel(dest.segment, this.segment);
        if (res.equals(MemorySegment.NULL)) {
            throw new MjException("Failed to copy mjModel into destination");
        }
    }

    /**
     * Returns the total byte size required to serialize this model using {@link #saveModelToBuffer()}.
     */
    public int sizeModel() {
        checkOpen();
        return Math.toIntExact(mujoco_h.mj_sizeModel(this.segment));
    }

    /**
     * Saves this model in binary MJB format to the specified file path.
     */
    public void saveModel(Path path) {
        checkOpen();
        Objects.requireNonNull(path, "path must not be null");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment pathSeg = temp.allocateFrom(path.toString());
            mujoco_h.mj_saveModel(this.segment, pathSeg, MemorySegment.NULL, 0);
        }
    }

    /**
     * Serializes this model in binary MJB format to a Java {@code byte[]}.
     */
    public byte[] saveModelToBuffer() {
        checkOpen();
        int size = sizeModel();
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment buf = temp.allocate(size);
            mujoco_h.mj_saveModel(this.segment, MemorySegment.NULL, buf, size);
            return buf.toArray(ValueLayout.JAVA_BYTE);
        }
    }

    /**
     * Saves the last parsed XML to the specified file path.
     */
    public void saveLastXml(Path path) {
        checkOpen();
        Objects.requireNonNull(path, "path must not be null");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment pathSeg = temp.allocateFrom(path.toString());
            MemorySegment errBuf = temp.allocate(1024);
            int ok = mujoco_h.mj_saveLastXML(pathSeg, this.segment, errBuf, 1024);
            if (ok == 0) {
                throw new MjException("Failed to save last XML: " + errBuf.getString(0));
            }
        }
    }

    /**
     * Prints a human-readable summary of this model to the specified file path.
     */
    public void printModel(Path path) {
        checkOpen();
        Objects.requireNonNull(path, "path must not be null");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment pathSeg = temp.allocateFrom(path.toString());
            mujoco_h.mj_printModel(this.segment, pathSeg);
        }
    }

    /**
     * Prints a human-readable summary of this model with custom float formatting to a file path.
     */
    public void printFormattedModel(Path path, String floatFormat) {
        checkOpen();
        Objects.requireNonNull(path, "path must not be null");
        Objects.requireNonNull(floatFormat, "floatFormat must not be null");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment pathSeg = temp.allocateFrom(path.toString());
            MemorySegment fmtSeg = temp.allocateFrom(floatFormat);
            mujoco_h.mj_printFormattedModel(this.segment, pathSeg, fmtSeg);
        }
    }

    /**
     * Computes the sum of all body masses in the model (`mj_getTotalmass`).
     */
    public double totalMass() {
        checkOpen();
        return mujoco_h.mj_getTotalmass(this.segment);
    }

    /**
     * Scales body masses and inertias so the total model mass equals {@code newMass} (`mj_setTotalmass`).
     */
    public void setTotalMass(double newMass) {
        checkOpen();
        mujoco_h.mj_setTotalmass(this.segment, newMass);
    }

    /**
     * Recomputes constant fields of the model based on {@code qpos0} (`mj_setConst`).
     */
    public void setConst(MjData data) {
        checkOpen();
        Objects.requireNonNull(data, "data must not be null");
        data.checkOpen();
        mujoco_h.mj_setConst(this.segment, data.segment());
    }

    /**
     * Computes actuator length range for the specified actuator index (`mj_setLengthRange`).
     */
    public void setLengthRange(MjData data, int index, MjLROpt opt) {
        checkOpen();
        Objects.requireNonNull(data, "data must not be null");
        Objects.requireNonNull(opt, "opt must not be null");
        data.checkOpen();
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment errBuf = temp.allocate(1024);
            int ok = mujoco_h.mj_setLengthRange(this.segment, data.segment(), index, opt.segment(), errBuf, 1024);
            if (ok == 0) {
                throw new MjException("mj_setLengthRange failed: " + errBuf.getString(0));
            }
        }
    }

    /**
     * Returns the raw memory address of the native {@code mjModel} struct.
     */
    public long address() {
        checkOpen();
        return segment.address();
    }

    MemorySegment segment() {
        checkOpen();
        return segment;
    }

    MemorySegment segmentUnchecked() {
        return segment;
    }

    public boolean isOpen() {
        return !closed.get() && arena.scope().isAlive();
    }

    void checkOpen() {
        if (!isOpen()) {
            throw new IllegalStateException("MjModel has already been closed");
        }
    }

    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            MjCallbacks.unregisterModel(this);
            arena.close();
        }
    }

    // =========================================================================
    // Sub-structs (opt, stat, vis)
    // =========================================================================

    public MjOption opt() {
        checkOpen();
        return optView;
    }

    public void setOpt(MjOption other) {
        checkOpen();
        Objects.requireNonNull(other, "other must not be null");
        optView.copyFrom(other);
    }

    public MjStatistic stat() {
        checkOpen();
        return statView;
    }

    public void setStat(MjStatistic other) {
        checkOpen();
        Objects.requireNonNull(other, "other must not be null");
        statView.copyFrom(other);
    }

    public MjVisual vis() {
        checkOpen();
        return visView;
    }

    public void setVis(MjVisual other) {
        checkOpen();
        Objects.requireNonNull(other, "other must not be null");
        visView.copyFrom(other);
    }

    public boolean flgGravcomp() {
        checkOpen();
        return mjModel_.flg_gravcomp(segment);
    }

    public void setFlgGravcomp(boolean val) {
        checkOpen();
        mjModel_.flg_gravcomp(segment, val);
    }

    public boolean flgSurfacevel() {
        checkOpen();
        return mjModel_.flg_surfacevel(segment);
    }

    public void setFlgSurfacevel(boolean val) {
        checkOpen();
        mjModel_.flg_surfacevel(segment, val);
    }

    // =========================================================================
    // Model Dimensions (mjtSize -> long / int convenience)
    // =========================================================================

    public int nq() { checkOpen(); return (int) mjModel_.nq(segment); }
    public int nv() { checkOpen(); return (int) mjModel_.nv(segment); }
    public int nu() { checkOpen(); return (int) mjModel_.nu(segment); }
    public int nactuator() { checkOpen(); return (int) mjModel_.nactuator(segment); }
    public int nout() { checkOpen(); return (int) mjModel_.nout(segment); }
    public int na() { checkOpen(); return (int) mjModel_.na(segment); }
    public int nbody() { checkOpen(); return (int) mjModel_.nbody(segment); }
    public int nbvh() { checkOpen(); return (int) mjModel_.nbvh(segment); }
    public int nbvhstatic() { checkOpen(); return (int) mjModel_.nbvhstatic(segment); }
    public int nbvhdynamic() { checkOpen(); return (int) mjModel_.nbvhdynamic(segment); }
    public int noct() { checkOpen(); return (int) mjModel_.noct(segment); }
    public int njnt() { checkOpen(); return (int) mjModel_.njnt(segment); }
    public int ntree() { checkOpen(); return (int) mjModel_.ntree(segment); }
    public int nM() { checkOpen(); return (int) mjModel_.nM(segment); }
    public int nB() { checkOpen(); return (int) mjModel_.nB(segment); }
    public int nC() { checkOpen(); return (int) mjModel_.nC(segment); }
    public int nD() { checkOpen(); return (int) mjModel_.nD(segment); }
    public int ngeom() { checkOpen(); return (int) mjModel_.ngeom(segment); }
    public int nsite() { checkOpen(); return (int) mjModel_.nsite(segment); }
    public int ncam() { checkOpen(); return (int) mjModel_.ncam(segment); }
    public int nlight() { checkOpen(); return (int) mjModel_.nlight(segment); }
    public int nflex() { checkOpen(); return (int) mjModel_.nflex(segment); }
    public int nmesh() { checkOpen(); return (int) mjModel_.nmesh(segment); }
    public int nskin() { checkOpen(); return (int) mjModel_.nskin(segment); }
    public int nhfield() { checkOpen(); return (int) mjModel_.nhfield(segment); }
    public int ntex() { checkOpen(); return (int) mjModel_.ntex(segment); }
    public long ntexdata() { checkOpen(); return mjModel_.ntexdata(segment); }
    public int nmat() { checkOpen(); return (int) mjModel_.nmat(segment); }
    public int npair() { checkOpen(); return (int) mjModel_.npair(segment); }
    public int nexclude() { checkOpen(); return (int) mjModel_.nexclude(segment); }
    public int neq() { checkOpen(); return (int) mjModel_.neq(segment); }
    public int ntendon() { checkOpen(); return (int) mjModel_.ntendon(segment); }
    public int nwrap() { checkOpen(); return (int) mjModel_.nwrap(segment); }
    public int nsensor() { checkOpen(); return (int) mjModel_.nsensor(segment); }
    public int nnumeric() { checkOpen(); return (int) mjModel_.nnumeric(segment); }
    public int ntext() { checkOpen(); return (int) mjModel_.ntext(segment); }
    public int ntuple() { checkOpen(); return (int) mjModel_.ntuple(segment); }
    public int nkey() { checkOpen(); return (int) mjModel_.nkey(segment); }
    public int nmocap() { checkOpen(); return (int) mjModel_.nmocap(segment); }
    public int nplugin() { checkOpen(); return (int) mjModel_.nplugin(segment); }
    public int nuserBody() { checkOpen(); return (int) mjModel_.nuser_body(segment); }
    public int nuserJnt() { checkOpen(); return (int) mjModel_.nuser_jnt(segment); }
    public int nuserGeom() { checkOpen(); return (int) mjModel_.nuser_geom(segment); }
    public int nuserSite() { checkOpen(); return (int) mjModel_.nuser_site(segment); }
    public int nuserCam() { checkOpen(); return (int) mjModel_.nuser_cam(segment); }
    public int nuserTendon() { checkOpen(); return (int) mjModel_.nuser_tendon(segment); }
    public int nuserActuator() { checkOpen(); return (int) mjModel_.nuser_actuator(segment); }
    public int nuserSensor() { checkOpen(); return (int) mjModel_.nuser_sensor(segment); }
    public int nnames() { checkOpen(); return (int) mjModel_.nnames(segment); }
    public int npaths() { checkOpen(); return (int) mjModel_.npaths(segment); }
    public int nnamesMap() { checkOpen(); return (int) mjModel_.nnames_map(segment); }
    public int nJmom() { checkOpen(); return (int) mjModel_.nJmom(segment); }
    public int ngravcomp() { checkOpen(); return (int) mjModel_.ngravcomp(segment); }
    public int nemax() { checkOpen(); return (int) mjModel_.nemax(segment); }
    public int njmax() { checkOpen(); return (int) mjModel_.njmax(segment); }
    public int nconmax() { checkOpen(); return (int) mjModel_.nconmax(segment); }
    public int nuserdata() { checkOpen(); return (int) mjModel_.nuserdata(segment); }
    public int nsensordata() { checkOpen(); return (int) mjModel_.nsensordata(segment); }
    public int npluginstate() { checkOpen(); return (int) mjModel_.npluginstate(segment); }
    public int nhistory() { checkOpen(); return (int) mjModel_.nhistory(segment); }
    public long narena() { checkOpen(); return mjModel_.narena(segment); }
    public long nbuffer() { checkOpen(); return mjModel_.nbuffer(segment); }

    // =========================================================================
    // Name / ID Lookups
    // =========================================================================

    /**
     * Looks up the integer ID of an object with the given type and name (`mj_name2id`).
     * Returns {@code -1} if no such object exists.
     */
    public int name2id(MjObjType type, String name) {
        Objects.requireNonNull(type, "type must not be null");
        return name2id(type.code(), name);
    }

    /**
     * Looks up the integer ID of an object with the given raw type code and name (`mj_name2id`).
     * Returns {@code -1} if no such object exists.
     */
    public int name2id(int typeCode, String name) {
        checkOpen();
        Objects.requireNonNull(name, "name must not be null");
        try (Arena temp = Arena.ofConfined()) {
            MemorySegment nameSeg = temp.allocateFrom(name);
            return mujoco_h.mj_name2id(this.segment, typeCode, nameSeg);
        }
    }

    /**
     * Looks up the name of an object with the given type and ID (`mj_id2name`).
     * Returns {@code null} if the object is unnamed or does not exist.
     */
    public String id2name(MjObjType type, int id) {
        Objects.requireNonNull(type, "type must not be null");
        return id2name(type.code(), id);
    }

    /**
     * Looks up the name of an object with the given raw type code and ID (`mj_id2name`).
     * Returns {@code null} if the object is unnamed or does not exist.
     */
    public String id2name(int typeCode, int id) {
        checkOpen();
        MemorySegment strPtr = mujoco_h.mj_id2name(this.segment, typeCode, id);
        if (strPtr.equals(MemorySegment.NULL)) {
            return null;
        }
        return strPtr.reinterpret(Long.MAX_VALUE).getString(0);
    }

    /**
     * Returns the raw {@code names} buffer as a byte vector of length {@link #nnames()}.
     */
    public MjByteVector namesBytes() {
        checkOpen();
        return new MjByteVector(mjModel_.names(segment), nnames(), this::isOpen);
    }

    /**
     * Reads a null-terminated UTF-8 string starting at {@code adr} inside the model's {@code names} table.
     */
    public String nameAt(int adr) {
        checkOpen();
        int total = nnames();
        if (adr < 0 || adr >= total) {
            return "";
        }
        MemorySegment namesSeg = mjModel_.names(segment).reinterpret(total);
        return namesSeg.getString(adr);
    }

    public String modelName() {
        return nameAt(0);
    }

    public int ntextdata() { checkOpen(); return Math.toIntExact(mjModel_.ntextdata(segment)); }
    public int nnumericdata() { checkOpen(); return Math.toIntExact(mjModel_.nnumericdata(segment)); }
    public MjIntVector textAdr() { checkOpen(); return intVec(mjModel_.text_adr(segment), ntext()); }
    public MjIntVector textSize() { checkOpen(); return intVec(mjModel_.text_size(segment), ntext()); }
    public MjByteVector textData() { checkOpen(); return byteVec(mjModel_.text_data(segment), ntextdata()); }
    public MjIntVector numericAdr() { checkOpen(); return intVec(mjModel_.numeric_adr(segment), nnumeric()); }
    public MjIntVector numericSize() { checkOpen(); return intVec(mjModel_.numeric_size(segment), nnumeric()); }
    public MjDoubleVector numericData() { checkOpen(); return doubleVec(mjModel_.numeric_data(segment), nnumericdata()); }

    public String textDataString(int textId) {
        checkOpen();
        if (textId < 0 || textId >= ntext()) {
            throw new IndexOutOfBoundsException("Invalid text id: " + textId);
        }
        int adr = textAdr().get(textId);
        return textData().getNullTerminatedString(adr);
    }

    // =========================================================================
    // Helper factory methods for zero-copy vector views
    // =========================================================================

    private MjDoubleVector doubleVec(MemorySegment ptr, long length) {
        return new MjDoubleVector(ptr, (int) length, this::isOpen);
    }

    private MjIntVector intVec(MemorySegment ptr, long length) {
        return new MjIntVector(ptr, (int) length, this::isOpen);
    }

    private MjLongVector longVec(MemorySegment ptr, long length) {
        return new MjLongVector(ptr, (int) length, this::isOpen);
    }

    private MjFloatVector floatVec(MemorySegment ptr, long length) {
        return new MjFloatVector(ptr, (int) length, this::isOpen);
    }

    private MjByteVector byteVec(MemorySegment ptr, long length) {
        return new MjByteVector(ptr, (int) length, this::isOpen);
    }

    // =========================================================================
    // Model Array Views (Zero-Copy)
    // =========================================================================

    // qpos0 & reference
    public MjDoubleVector qpos0() { checkOpen(); return doubleVec(mjModel_.qpos0(segment), nq()); }
    public MjDoubleVector qposSpring() { checkOpen(); return doubleVec(mjModel_.qpos_spring(segment), nq()); }

    // Bodies (nbody)
    public MjIntVector bodyParentid() { checkOpen(); return intVec(mjModel_.body_parentid(segment), nbody()); }
    public MjIntVector bodyRootid() { checkOpen(); return intVec(mjModel_.body_rootid(segment), nbody()); }
    public MjIntVector bodyWeldid() { checkOpen(); return intVec(mjModel_.body_weldid(segment), nbody()); }
    public MjIntVector bodyMocapid() { checkOpen(); return intVec(mjModel_.body_mocapid(segment), nbody()); }
    public MjIntVector bodyJntnum() { checkOpen(); return intVec(mjModel_.body_jntnum(segment), nbody()); }
    public MjIntVector bodyJntadr() { checkOpen(); return intVec(mjModel_.body_jntadr(segment), nbody()); }
    public MjIntVector bodyDofnum() { checkOpen(); return intVec(mjModel_.body_dofnum(segment), nbody()); }
    public MjIntVector bodyDofadr() { checkOpen(); return intVec(mjModel_.body_dofadr(segment), nbody()); }
    public MjIntVector bodyTreeid() { checkOpen(); return intVec(mjModel_.body_treeid(segment), nbody()); }
    public MjIntVector bodyGeomnum() { checkOpen(); return intVec(mjModel_.body_geomnum(segment), nbody()); }
    public MjIntVector bodyGeomadr() { checkOpen(); return intVec(mjModel_.body_geomadr(segment), nbody()); }
    public MjByteVector bodySimple() { checkOpen(); return byteVec(mjModel_.body_simple(segment), nbody()); }
    public MjByteVector bodySameframe() { checkOpen(); return byteVec(mjModel_.body_sameframe(segment), nbody()); }
    public MjDoubleVector bodyPos() { checkOpen(); return doubleVec(mjModel_.body_pos(segment), 3L * nbody()); }
    public MjDoubleVector bodyQuat() { checkOpen(); return doubleVec(mjModel_.body_quat(segment), 4L * nbody()); }
    public MjDoubleVector bodyIpos() { checkOpen(); return doubleVec(mjModel_.body_ipos(segment), 3L * nbody()); }
    public MjDoubleVector bodyIquat() { checkOpen(); return doubleVec(mjModel_.body_iquat(segment), 4L * nbody()); }
    public MjDoubleVector bodyMass() { checkOpen(); return doubleVec(mjModel_.body_mass(segment), nbody()); }
    public MjDoubleVector bodySubtreemass() { checkOpen(); return doubleVec(mjModel_.body_subtreemass(segment), nbody()); }
    public MjDoubleVector bodyInertia() { checkOpen(); return doubleVec(mjModel_.body_inertia(segment), 3L * nbody()); }
    public MjDoubleVector bodyInvweight0() { checkOpen(); return doubleVec(mjModel_.body_invweight0(segment), 2L * nbody()); }
    public MjDoubleVector bodyGravcomp() { checkOpen(); return doubleVec(mjModel_.body_gravcomp(segment), nbody()); }
    public MjDoubleVector bodyMargin() { checkOpen(); return doubleVec(mjModel_.body_margin(segment), nbody()); }
    public MjDoubleVector bodyUser() { checkOpen(); return doubleVec(mjModel_.body_user(segment), (long) nbody() * nuserBody()); }

    // Joints (njnt)
    public MjIntVector jntType() { checkOpen(); return intVec(mjModel_.jnt_type(segment), njnt()); }
    public MjIntVector jntQposadr() { checkOpen(); return intVec(mjModel_.jnt_qposadr(segment), njnt()); }
    public MjIntVector jntDofadr() { checkOpen(); return intVec(mjModel_.jnt_dofadr(segment), njnt()); }
    public MjIntVector jntBodyid() { checkOpen(); return intVec(mjModel_.jnt_bodyid(segment), njnt()); }
    public MjIntVector jntGroup() { checkOpen(); return intVec(mjModel_.jnt_group(segment), njnt()); }
    public MjByteVector jntLimited() { checkOpen(); return byteVec(mjModel_.jnt_limited(segment), njnt()); }
    public MjByteVector jntActfrclimited() { checkOpen(); return byteVec(mjModel_.jnt_actfrclimited(segment), njnt()); }
    public MjByteVector jntActgravcomp() { checkOpen(); return byteVec(mjModel_.jnt_actgravcomp(segment), njnt()); }
    public MjDoubleVector jntSolref() { checkOpen(); return doubleVec(mjModel_.jnt_solref(segment), (long) MjConstants.mjNREF * njnt()); }
    public MjDoubleVector jntSolimp() { checkOpen(); return doubleVec(mjModel_.jnt_solimp(segment), (long) MjConstants.mjNIMP * njnt()); }
    public MjDoubleVector jntPos() { checkOpen(); return doubleVec(mjModel_.jnt_pos(segment), 3L * njnt()); }
    public MjDoubleVector jntAxis() { checkOpen(); return doubleVec(mjModel_.jnt_axis(segment), 3L * njnt()); }
    public MjDoubleVector jntStiffness() { checkOpen(); return doubleVec(mjModel_.jnt_stiffness(segment), njnt()); }
    public MjDoubleVector jntStiffnesspoly() { checkOpen(); return doubleVec(mjModel_.jnt_stiffnesspoly(segment), (long) MjConstants.mjNPOLY * njnt()); }
    public MjDoubleVector jntRange() { checkOpen(); return doubleVec(mjModel_.jnt_range(segment), 2L * njnt()); }
    public MjDoubleVector jntActfrcrange() { checkOpen(); return doubleVec(mjModel_.jnt_actfrcrange(segment), 2L * njnt()); }
    public MjDoubleVector jntMargin() { checkOpen(); return doubleVec(mjModel_.jnt_margin(segment), njnt()); }
    public MjDoubleVector jntUser() { checkOpen(); return doubleVec(mjModel_.jnt_user(segment), (long) njnt() * nuserJnt()); }

    // DOFs (nv)
    public MjIntVector dofBodyid() { checkOpen(); return intVec(mjModel_.dof_bodyid(segment), nv()); }
    public MjIntVector dofJntid() { checkOpen(); return intVec(mjModel_.dof_jntid(segment), nv()); }
    public MjIntVector dofParentid() { checkOpen(); return intVec(mjModel_.dof_parentid(segment), nv()); }
    public MjIntVector dofTreeid() { checkOpen(); return intVec(mjModel_.dof_treeid(segment), nv()); }
    public MjIntVector dofMadr() { checkOpen(); return intVec(mjModel_.dof_Madr(segment), nv()); }
    public MjIntVector dofSimplenum() { checkOpen(); return intVec(mjModel_.dof_simplenum(segment), nv()); }
    public MjDoubleVector dofSolref() { checkOpen(); return doubleVec(mjModel_.dof_solref(segment), (long) MjConstants.mjNREF * nv()); }
    public MjDoubleVector dofSolimp() { checkOpen(); return doubleVec(mjModel_.dof_solimp(segment), (long) MjConstants.mjNIMP * nv()); }
    public MjDoubleVector dofFrictionloss() { checkOpen(); return doubleVec(mjModel_.dof_frictionloss(segment), nv()); }
    public MjDoubleVector dofArmature() { checkOpen(); return doubleVec(mjModel_.dof_armature(segment), nv()); }
    public MjDoubleVector dofDamping() { checkOpen(); return doubleVec(mjModel_.dof_damping(segment), nv()); }
    public MjDoubleVector dofDampingpoly() { checkOpen(); return doubleVec(mjModel_.dof_dampingpoly(segment), (long) MjConstants.mjNPOLY * nv()); }
    public MjDoubleVector dofInvweight0() { checkOpen(); return doubleVec(mjModel_.dof_invweight0(segment), nv()); }
    public MjDoubleVector dofM0() { checkOpen(); return doubleVec(mjModel_.dof_M0(segment), nv()); }

    // Geoms (ngeom)
    public MjIntVector geomType() { checkOpen(); return intVec(mjModel_.geom_type(segment), ngeom()); }
    public MjIntVector geomContype() { checkOpen(); return intVec(mjModel_.geom_contype(segment), ngeom()); }
    public MjIntVector geomConaffinity() { checkOpen(); return intVec(mjModel_.geom_conaffinity(segment), ngeom()); }
    public MjIntVector geomCondim() { checkOpen(); return intVec(mjModel_.geom_condim(segment), ngeom()); }
    public MjIntVector geomBodyid() { checkOpen(); return intVec(mjModel_.geom_bodyid(segment), ngeom()); }
    public MjIntVector geomDataid() { checkOpen(); return intVec(mjModel_.geom_dataid(segment), ngeom()); }
    public MjIntVector geomMatid() { checkOpen(); return intVec(mjModel_.geom_matid(segment), ngeom()); }
    public MjIntVector geomGroup() { checkOpen(); return intVec(mjModel_.geom_group(segment), ngeom()); }
    public MjIntVector geomPriority() { checkOpen(); return intVec(mjModel_.geom_priority(segment), ngeom()); }
    public MjIntVector geomPlugin() { checkOpen(); return intVec(mjModel_.geom_plugin(segment), ngeom()); }
    public MjByteVector geomSameframe() { checkOpen(); return byteVec(mjModel_.geom_sameframe(segment), ngeom()); }
    public MjDoubleVector geomSolmix() { checkOpen(); return doubleVec(mjModel_.geom_solmix(segment), ngeom()); }
    public MjDoubleVector geomSolref() { checkOpen(); return doubleVec(mjModel_.geom_solref(segment), (long) MjConstants.mjNREF * ngeom()); }
    public MjDoubleVector geomSolimp() { checkOpen(); return doubleVec(mjModel_.geom_solimp(segment), (long) MjConstants.mjNIMP * ngeom()); }
    public MjDoubleVector geomSize() { checkOpen(); return doubleVec(mjModel_.geom_size(segment), 3L * ngeom()); }
    public MjDoubleVector geomAabb() { checkOpen(); return doubleVec(mjModel_.geom_aabb(segment), 6L * ngeom()); }
    public MjDoubleVector geomRbound() { checkOpen(); return doubleVec(mjModel_.geom_rbound(segment), ngeom()); }
    public MjDoubleVector geomPos() { checkOpen(); return doubleVec(mjModel_.geom_pos(segment), 3L * ngeom()); }
    public MjDoubleVector geomQuat() { checkOpen(); return doubleVec(mjModel_.geom_quat(segment), 4L * ngeom()); }
    public MjDoubleVector geomFriction() { checkOpen(); return doubleVec(mjModel_.geom_friction(segment), 3L * ngeom()); }
    public MjDoubleVector geomMargin() { checkOpen(); return doubleVec(mjModel_.geom_margin(segment), ngeom()); }
    public MjDoubleVector geomGap() { checkOpen(); return doubleVec(mjModel_.geom_gap(segment), ngeom()); }
    public MjDoubleVector geomFluid() { checkOpen(); return doubleVec(mjModel_.geom_fluid(segment), (long) MjConstants.mjNFLUID * ngeom()); }
    public MjDoubleVector geomUser() { checkOpen(); return doubleVec(mjModel_.geom_user(segment), (long) ngeom() * nuserGeom()); }
    public MjFloatVector geomRgba() { checkOpen(); return floatVec(mjModel_.geom_rgba(segment), 4L * ngeom()); }

    // Sites (nsite)
    public MjIntVector siteType() { checkOpen(); return intVec(mjModel_.site_type(segment), nsite()); }
    public MjIntVector siteBodyid() { checkOpen(); return intVec(mjModel_.site_bodyid(segment), nsite()); }
    public MjIntVector siteMatid() { checkOpen(); return intVec(mjModel_.site_matid(segment), nsite()); }
    public MjIntVector siteGroup() { checkOpen(); return intVec(mjModel_.site_group(segment), nsite()); }
    public MjByteVector siteSameframe() { checkOpen(); return byteVec(mjModel_.site_sameframe(segment), nsite()); }
    public MjDoubleVector siteSize() { checkOpen(); return doubleVec(mjModel_.site_size(segment), 3L * nsite()); }
    public MjDoubleVector sitePos() { checkOpen(); return doubleVec(mjModel_.site_pos(segment), 3L * nsite()); }
    public MjDoubleVector siteQuat() { checkOpen(); return doubleVec(mjModel_.site_quat(segment), 4L * nsite()); }
    public MjDoubleVector siteUser() { checkOpen(); return doubleVec(mjModel_.site_user(segment), (long) nsite() * nuserSite()); }
    public MjFloatVector siteRgba() { checkOpen(); return floatVec(mjModel_.site_rgba(segment), 4L * nsite()); }

    // Cameras (ncam)
    public MjIntVector camMode() { checkOpen(); return intVec(mjModel_.cam_mode(segment), ncam()); }
    public MjIntVector camBodyid() { checkOpen(); return intVec(mjModel_.cam_bodyid(segment), ncam()); }
    public MjIntVector camTargetbodyid() { checkOpen(); return intVec(mjModel_.cam_targetbodyid(segment), ncam()); }
    public MjDoubleVector camPos() { checkOpen(); return doubleVec(mjModel_.cam_pos(segment), 3L * ncam()); }
    public MjDoubleVector camQuat() { checkOpen(); return doubleVec(mjModel_.cam_quat(segment), 4L * ncam()); }
    public MjDoubleVector camPoscom0() { checkOpen(); return doubleVec(mjModel_.cam_poscom0(segment), 3L * ncam()); }
    public MjDoubleVector camPos0() { checkOpen(); return doubleVec(mjModel_.cam_pos0(segment), 3L * ncam()); }
    public MjDoubleVector camMat0() { checkOpen(); return doubleVec(mjModel_.cam_mat0(segment), 9L * ncam()); }
    public MjDoubleVector camFovy() { checkOpen(); return doubleVec(mjModel_.cam_fovy(segment), ncam()); }
    public MjDoubleVector camIpd() { checkOpen(); return doubleVec(mjModel_.cam_ipd(segment), ncam()); }
    public MjIntVector camResolution() { checkOpen(); return intVec(mjModel_.cam_resolution(segment), 2L * ncam()); }
    public MjFloatVector camSensorsize() { checkOpen(); return floatVec(mjModel_.cam_sensorsize(segment), 2L * ncam()); }
    public MjFloatVector camIntrinsic() { checkOpen(); return floatVec(mjModel_.cam_intrinsic(segment), 4L * ncam()); }

    // Actuators (nactuator)
    public MjIntVector actuatorTrntype() { checkOpen(); return intVec(mjModel_.actuator_trntype(segment), nactuator()); }
    public MjIntVector actuatorDyntype() { checkOpen(); return intVec(mjModel_.actuator_dyntype(segment), nactuator()); }
    public MjIntVector actuatorGaintype() { checkOpen(); return intVec(mjModel_.actuator_gaintype(segment), nactuator()); }
    public MjIntVector actuatorBiastype() { checkOpen(); return intVec(mjModel_.actuator_biastype(segment), nactuator()); }
    public MjIntVector actuatorCtrladr() { checkOpen(); return intVec(mjModel_.actuator_ctrladr(segment), nactuator()); }
    public MjIntVector actuatorCtrlnum() { checkOpen(); return intVec(mjModel_.actuator_ctrlnum(segment), nactuator()); }
    public MjIntVector actuatorOutadr() { checkOpen(); return intVec(mjModel_.actuator_outadr(segment), nactuator()); }
    public MjIntVector actuatorOutnum() { checkOpen(); return intVec(mjModel_.actuator_outnum(segment), nactuator()); }
    public MjIntVector actuatorTrnid() { checkOpen(); return intVec(mjModel_.actuator_trnid(segment), 2L * nactuator()); }
    public MjIntVector actuatorActadr() { checkOpen(); return intVec(mjModel_.actuator_actadr(segment), nactuator()); }
    public MjIntVector actuatorActnum() { checkOpen(); return intVec(mjModel_.actuator_actnum(segment), nactuator()); }
    public MjIntVector actuatorGroup() { checkOpen(); return intVec(mjModel_.actuator_group(segment), nactuator()); }
    public MjIntVector actuatorHistory() { checkOpen(); return intVec(mjModel_.actuator_history(segment), 2L * nactuator()); }
    public MjIntVector actuatorHistoryadr() { checkOpen(); return intVec(mjModel_.actuator_historyadr(segment), nactuator()); }
    public MjDoubleVector actuatorDelay() { checkOpen(); return doubleVec(mjModel_.actuator_delay(segment), nactuator()); }
    public MjByteVector actuatorCtrllimited() { checkOpen(); return byteVec(mjModel_.actuator_ctrllimited(segment), nactuator()); }
    public MjByteVector actuatorForcelimited() { checkOpen(); return byteVec(mjModel_.actuator_forcelimited(segment), nactuator()); }
    public MjByteVector actuatorActlimited() { checkOpen(); return byteVec(mjModel_.actuator_actlimited(segment), nactuator()); }
    public MjDoubleVector actuatorDynprm() { checkOpen(); return doubleVec(mjModel_.actuator_dynprm(segment), (long) MjConstants.mjNDYN * nactuator()); }
    public MjDoubleVector actuatorGainprm() { checkOpen(); return doubleVec(mjModel_.actuator_gainprm(segment), (long) MjConstants.mjNGAIN * nactuator()); }
    public MjDoubleVector actuatorBiasprm() { checkOpen(); return doubleVec(mjModel_.actuator_biasprm(segment), (long) MjConstants.mjNBIAS * nactuator()); }
    public MjByteVector actuatorActearly() { checkOpen(); return byteVec(mjModel_.actuator_actearly(segment), nactuator()); }
    public MjDoubleVector actuatorCtrlrange() { checkOpen(); return doubleVec(mjModel_.actuator_ctrlrange(segment), 2L * nactuator()); }
    public MjDoubleVector actuatorForcerange() { checkOpen(); return doubleVec(mjModel_.actuator_forcerange(segment), 2L * nactuator()); }
    public MjDoubleVector actuatorActrange() { checkOpen(); return doubleVec(mjModel_.actuator_actrange(segment), 2L * nactuator()); }
    public MjDoubleVector actuatorGear() { checkOpen(); return doubleVec(mjModel_.actuator_gear(segment), 6L * nactuator()); }
    public MjDoubleVector actuatorCranklength() { checkOpen(); return doubleVec(mjModel_.actuator_cranklength(segment), nactuator()); }
    public MjDoubleVector actuatorAcc0() { checkOpen(); return doubleVec(mjModel_.actuator_acc0(segment), nactuator()); }
    public MjDoubleVector actuatorLength0() { checkOpen(); return doubleVec(mjModel_.actuator_length0(segment), nactuator()); }
    public MjDoubleVector actuatorLengthrange() { checkOpen(); return doubleVec(mjModel_.actuator_lengthrange(segment), 2L * nactuator()); }
    public MjDoubleVector actuatorUser() { checkOpen(); return doubleVec(mjModel_.actuator_user(segment), (long) nactuator() * nuserActuator()); }

    // Sensors (nsensor)
    public MjIntVector sensorType() { checkOpen(); return intVec(mjModel_.sensor_type(segment), nsensor()); }
    public MjIntVector sensorDatatype() { checkOpen(); return intVec(mjModel_.sensor_datatype(segment), nsensor()); }
    public MjIntVector sensorNeedstage() { checkOpen(); return intVec(mjModel_.sensor_needstage(segment), nsensor()); }
    public MjIntVector sensorObjtype() { checkOpen(); return intVec(mjModel_.sensor_objtype(segment), nsensor()); }
    public MjIntVector sensorObjid() { checkOpen(); return intVec(mjModel_.sensor_objid(segment), nsensor()); }
    public MjIntVector sensorReftype() { checkOpen(); return intVec(mjModel_.sensor_reftype(segment), nsensor()); }
    public MjIntVector sensorRefid() { checkOpen(); return intVec(mjModel_.sensor_refid(segment), nsensor()); }
    public MjIntVector sensorIntprm() { checkOpen(); return intVec(mjModel_.sensor_intprm(segment), (long) MjConstants.mjNSENS * nsensor()); }
    public MjIntVector sensorDim() { checkOpen(); return intVec(mjModel_.sensor_dim(segment), nsensor()); }
    public MjIntVector sensorAdr() { checkOpen(); return intVec(mjModel_.sensor_adr(segment), nsensor()); }
    public MjDoubleVector sensorCutoff() { checkOpen(); return doubleVec(mjModel_.sensor_cutoff(segment), nsensor()); }
    public MjDoubleVector sensorNoise() { checkOpen(); return doubleVec(mjModel_.sensor_noise(segment), nsensor()); }
    public MjIntVector sensorHistory() { checkOpen(); return intVec(mjModel_.sensor_history(segment), 2L * nsensor()); }
    public MjIntVector sensorHistoryadr() { checkOpen(); return intVec(mjModel_.sensor_historyadr(segment), nsensor()); }
    public MjDoubleVector sensorDelay() { checkOpen(); return doubleVec(mjModel_.sensor_delay(segment), nsensor()); }
    public MjDoubleVector sensorInterval() { checkOpen(); return doubleVec(mjModel_.sensor_interval(segment), 2L * nsensor()); }
    public MjDoubleVector sensorUser() { checkOpen(); return doubleVec(mjModel_.sensor_user(segment), (long) nsensor() * nuserSensor()); }

    // Tendons (ntendon)
    public MjIntVector tendonAdr() { checkOpen(); return intVec(mjModel_.tendon_adr(segment), ntendon()); }
    public MjIntVector tendonNum() { checkOpen(); return intVec(mjModel_.tendon_num(segment), ntendon()); }
    public MjIntVector tendonMatid() { checkOpen(); return intVec(mjModel_.tendon_matid(segment), ntendon()); }
    public MjIntVector tendonGroup() { checkOpen(); return intVec(mjModel_.tendon_group(segment), ntendon()); }
    public MjByteVector tendonLimited() { checkOpen(); return byteVec(mjModel_.tendon_limited(segment), ntendon()); }
    public MjDoubleVector tendonWidth() { checkOpen(); return doubleVec(mjModel_.tendon_width(segment), ntendon()); }
    public MjDoubleVector tendonStiffness() { checkOpen(); return doubleVec(mjModel_.tendon_stiffness(segment), ntendon()); }
    public MjDoubleVector tendonDamping() { checkOpen(); return doubleVec(mjModel_.tendon_damping(segment), ntendon()); }
    public MjDoubleVector tendonRange() { checkOpen(); return doubleVec(mjModel_.tendon_range(segment), 2L * ntendon()); }
    public MjDoubleVector tendonLength0() { checkOpen(); return doubleVec(mjModel_.tendon_length0(segment), ntendon()); }

    // Equality constraints (neq)
    public MjIntVector eqType() { checkOpen(); return intVec(mjModel_.eq_type(segment), neq()); }
    public MjIntVector eqObj1id() { checkOpen(); return intVec(mjModel_.eq_obj1id(segment), neq()); }
    public MjIntVector eqObj2id() { checkOpen(); return intVec(mjModel_.eq_obj2id(segment), neq()); }
    public MjIntVector eqObjtype() { checkOpen(); return intVec(mjModel_.eq_objtype(segment), neq()); }
    public MjByteVector eqActive0() { checkOpen(); return byteVec(mjModel_.eq_active0(segment), neq()); }
    public MjDoubleVector eqSolref() { checkOpen(); return doubleVec(mjModel_.eq_solref(segment), (long) MjConstants.mjNREF * neq()); }
    public MjDoubleVector eqSolimp() { checkOpen(); return doubleVec(mjModel_.eq_solimp(segment), (long) MjConstants.mjNIMP * neq()); }
    public MjDoubleVector eqData() { checkOpen(); return doubleVec(mjModel_.eq_data(segment), (long) MjConstants.mjNEQDATA * neq()); }

    // Textures (ntex)
    public MjIntVector texType() { checkOpen(); return intVec(mjModel_.tex_type(segment), ntex()); }
    public MjIntVector texColorspace() { checkOpen(); return intVec(mjModel_.tex_colorspace(segment), ntex()); }
    public MjIntVector texHeight() { checkOpen(); return intVec(mjModel_.tex_height(segment), ntex()); }
    public MjIntVector texWidth() { checkOpen(); return intVec(mjModel_.tex_width(segment), ntex()); }
    public MjIntVector texNchannel() { checkOpen(); return intVec(mjModel_.tex_nchannel(segment), ntex()); }
    public MjLongVector texAdr() { checkOpen(); return longVec(mjModel_.tex_adr(segment), ntex()); }
    public MjByteVector texData() { checkOpen(); return byteVec(mjModel_.tex_data(segment), ntexdata()); }

    // Keyframes (nkey)
    public MjDoubleVector keyTime() { checkOpen(); return doubleVec(mjModel_.key_time(segment), nkey()); }
    public MjDoubleVector keyQpos() { checkOpen(); return doubleVec(mjModel_.key_qpos(segment), (long) nkey() * nq()); }
    public MjDoubleVector keyQvel() { checkOpen(); return doubleVec(mjModel_.key_qvel(segment), (long) nkey() * nv()); }
    public MjDoubleVector keyAct() { checkOpen(); return doubleVec(mjModel_.key_act(segment), (long) nkey() * na()); }
    public MjDoubleVector keyMpos() { checkOpen(); return doubleVec(mjModel_.key_mpos(segment), 3L * nkey() * nmocap()); }
    public MjDoubleVector keyMquat() { checkOpen(); return doubleVec(mjModel_.key_mquat(segment), 4L * nkey() * nmocap()); }
    public MjDoubleVector keyCtrl() { checkOpen(); return doubleVec(mjModel_.key_ctrl(segment), (long) nkey() * nu()); }

    // Name address tables
    public MjIntVector nameBodyadr() { checkOpen(); return intVec(mjModel_.name_bodyadr(segment), nbody()); }
    public MjIntVector nameJntadr() { checkOpen(); return intVec(mjModel_.name_jntadr(segment), njnt()); }
    public MjIntVector nameGeomadr() { checkOpen(); return intVec(mjModel_.name_geomadr(segment), ngeom()); }
    public MjIntVector nameSiteadr() { checkOpen(); return intVec(mjModel_.name_siteadr(segment), nsite()); }
    public MjIntVector nameCamadr() { checkOpen(); return intVec(mjModel_.name_camadr(segment), ncam()); }
    public MjIntVector nameLightadr() { checkOpen(); return intVec(mjModel_.name_lightadr(segment), nlight()); }
    public MjIntVector nameTexadr() { checkOpen(); return intVec(mjModel_.name_texadr(segment), ntex()); }
    public MjIntVector nameMatadr() { checkOpen(); return intVec(mjModel_.name_matadr(segment), nmat()); }
    public MjIntVector nameEqadr() { checkOpen(); return intVec(mjModel_.name_eqadr(segment), neq()); }
    public MjIntVector nameTendonadr() { checkOpen(); return intVec(mjModel_.name_tendonadr(segment), ntendon()); }
    public MjIntVector nameActuatoradr() { checkOpen(); return intVec(mjModel_.name_actuatoradr(segment), nactuator()); }
    public MjIntVector nameSensoradr() { checkOpen(); return intVec(mjModel_.name_sensoradr(segment), nsensor()); }
    public MjIntVector nameKeyadr() { checkOpen(); return intVec(mjModel_.name_keyadr(segment), nkey()); }

    // =========================================================================
    // Named & Indexed Element Views (DeepMind bindings_test.py style)
    // =========================================================================

    public BodyView body(int id) {
        checkOpen();
        if (id < 0 || id >= nbody()) {
            throw new IndexOutOfBoundsException("Invalid body id " + id + " (nbody=" + nbody() + ")");
        }
        return new BodyView(id);
    }

    public BodyView body(String name) {
        checkOpen();
        int id = name2id(MjObjType.BODY, name);
        if (id < 0) {
            throw new IllegalArgumentException("Unknown body name: '" + name + "'");
        }
        return new BodyView(id);
    }

    public JointView joint(int id) {
        checkOpen();
        if (id < 0 || id >= njnt()) {
            throw new IndexOutOfBoundsException("Invalid joint id " + id + " (njnt=" + njnt() + ")");
        }
        return new JointView(id);
    }

    public JointView joint(String name) {
        checkOpen();
        int id = name2id(MjObjType.JOINT, name);
        if (id < 0) {
            throw new IllegalArgumentException("Unknown joint name: '" + name + "'");
        }
        return new JointView(id);
    }

    public GeomView geom(int id) {
        checkOpen();
        if (id < 0 || id >= ngeom()) {
            throw new IndexOutOfBoundsException("Invalid geom id " + id + " (ngeom=" + ngeom() + ")");
        }
        return new GeomView(id);
    }

    public GeomView geom(String name) {
        checkOpen();
        int id = name2id(MjObjType.GEOM, name);
        if (id < 0) {
            throw new IllegalArgumentException("Unknown geom name: '" + name + "'");
        }
        return new GeomView(id);
    }

    public SiteView site(int id) {
        checkOpen();
        if (id < 0 || id >= nsite()) {
            throw new IndexOutOfBoundsException("Invalid site id " + id + " (nsite=" + nsite() + ")");
        }
        return new SiteView(id);
    }

    public SiteView site(String name) {
        checkOpen();
        int id = name2id(MjObjType.SITE, name);
        if (id < 0) {
            throw new IllegalArgumentException("Unknown site name: '" + name + "'");
        }
        return new SiteView(id);
    }

    public ActuatorView actuator(int id) {
        checkOpen();
        if (id < 0 || id >= nactuator()) {
            throw new IndexOutOfBoundsException("Invalid actuator id " + id + " (nactuator=" + nactuator() + ")");
        }
        return new ActuatorView(id);
    }

    public ActuatorView actuator(String name) {
        checkOpen();
        int id = name2id(MjObjType.ACTUATOR, name);
        if (id < 0) {
            throw new IllegalArgumentException("Unknown actuator name: '" + name + "'");
        }
        return new ActuatorView(id);
    }

    public SensorView sensor(int id) {
        checkOpen();
        if (id < 0 || id >= nsensor()) {
            throw new IndexOutOfBoundsException("Invalid sensor id " + id + " (nsensor=" + nsensor() + ")");
        }
        return new SensorView(id);
    }

    public SensorView sensor(String name) {
        checkOpen();
        int id = name2id(MjObjType.SENSOR, name);
        if (id < 0) {
            throw new IllegalArgumentException("Unknown sensor name: '" + name + "'");
        }
        return new SensorView(id);
    }

    public TexView tex(int id) {
        checkOpen();
        if (id < 0 || id >= ntex()) {
            throw new IllegalArgumentException("Invalid texture id " + id + " (ntex=" + ntex() + ")");
        }
        return new TexView(id);
    }

    public TexView tex(String name) {
        checkOpen();
        int id = name2id(MjObjType.TEXTURE, name);
        if (id < 0) {
            throw new IllegalArgumentException("Unknown texture name: '" + name + "'");
        }
        return new TexView(id);
    }

    // =========================================================================
    // View Inner Classes
    // =========================================================================

    public final class BodyView {
        private final int id;
        private BodyView(int id) { this.id = id; }
        public int id() { checkOpen(); return id; }
        public String name() { checkOpen(); String n = id2name(MjObjType.BODY, id); return n != null ? n : ""; }
        public int parentid() { return bodyParentid().get(id); }
        public int rootid() { return bodyRootid().get(id); }
        public int weldid() { return bodyWeldid().get(id); }
        public int mocapid() { return bodyMocapid().get(id); }
        public int jntnum() { return bodyJntnum().get(id); }
        public int jntadr() { return bodyJntadr().get(id); }
        public int dofnum() { return bodyDofnum().get(id); }
        public int dofadr() { return bodyDofadr().get(id); }
        public int geomnum() { return bodyGeomnum().get(id); }
        public int geomadr() { return bodyGeomadr().get(id); }
        public MjDoubleVector pos() { return bodyPos().slice(3 * id, 3); }
        public MjDoubleVector quat() { return bodyQuat().slice(4 * id, 4); }
        public MjDoubleVector ipos() { return bodyIpos().slice(3 * id, 3); }
        public MjDoubleVector iquat() { return bodyIquat().slice(4 * id, 4); }
        public double mass() { return bodyMass().get(id); }
        public void setMass(double m) { bodyMass().set(id, m); }
        public double subtreemass() { return bodySubtreemass().get(id); }
        public MjDoubleVector inertia() { return bodyInertia().slice(3 * id, 3); }
        public double gravcomp() { return bodyGravcomp().get(id); }
    }

    public final class JointView {
        private final int id;
        private JointView(int id) { this.id = id; }
        public int id() { checkOpen(); return id; }
        public String name() { checkOpen(); String n = id2name(MjObjType.JOINT, id); return n != null ? n : ""; }
        public int type() { return jntType().get(id); }
        public int qposadr() { return jntQposadr().get(id); }
        public int dofadr() { return jntDofadr().get(id); }
        public int bodyid() { return jntBodyid().get(id); }
        public MjDoubleVector pos() { return jntPos().slice(3 * id, 3); }
        public MjDoubleVector axis() { return jntAxis().slice(3 * id, 3); }
        public MjDoubleVector stiffness() { return jntStiffness().slice(id, 1); }
        public MjDoubleVector stiffnesspoly() { return jntStiffnesspoly().slice(MjConstants.mjNPOLY * id, MjConstants.mjNPOLY); }
        public MjDoubleVector range() { return jntRange().slice(2 * id, 2); }
    }

    public final class GeomView {
        private final int id;
        private GeomView(int id) { this.id = id; }
        public int id() { checkOpen(); return id; }
        public String name() { checkOpen(); String n = id2name(MjObjType.GEOM, id); return n != null ? n : ""; }
        public int type() { return geomType().get(id); }
        public int bodyid() { return geomBodyid().get(id); }
        public MjDoubleVector size() { return geomSize().slice(3 * id, 3); }
        public MjDoubleVector pos() { return geomPos().slice(3 * id, 3); }
        public MjDoubleVector quat() { return geomQuat().slice(4 * id, 4); }
        public MjDoubleVector friction() { return geomFriction().slice(3 * id, 3); }
        public MjFloatVector rgba() { return geomRgba().slice(4 * id, 4); }
        public double rbound() { return geomRbound().get(id); }
    }

    public final class SiteView {
        private final int id;
        private SiteView(int id) { this.id = id; }
        public int id() { checkOpen(); return id; }
        public String name() { checkOpen(); String n = id2name(MjObjType.SITE, id); return n != null ? n : ""; }
        public int type() { return siteType().get(id); }
        public int bodyid() { return siteBodyid().get(id); }
        public MjDoubleVector size() { return siteSize().slice(3 * id, 3); }
        public MjDoubleVector pos() { return sitePos().slice(3 * id, 3); }
        public MjDoubleVector quat() { return siteQuat().slice(4 * id, 4); }
        public MjFloatVector rgba() { return siteRgba().slice(4 * id, 4); }
    }

    public final class ActuatorView {
        private final int id;
        private ActuatorView(int id) { this.id = id; }
        public int id() { checkOpen(); return id; }
        public String name() { checkOpen(); String n = id2name(MjObjType.ACTUATOR, id); return n != null ? n : ""; }
        public int trntype() { return actuatorTrntype().get(id); }
        public int dyntype() { return actuatorDyntype().get(id); }
        public int gaintype() { return actuatorGaintype().get(id); }
        public int biastype() { return actuatorBiastype().get(id); }
        public int ctrladr() { return actuatorCtrladr().get(id); }
        public int ctrlnum() { return actuatorCtrlnum().get(id); }
        public int outadr() { return actuatorOutadr().get(id); }
        public int outnum() { return actuatorOutnum().get(id); }
        public MjIntVector trnid() { return actuatorTrnid().slice(2 * id, 2); }
        public MjDoubleVector dynprm() { return actuatorDynprm().slice(MjConstants.mjNDYN * id, MjConstants.mjNDYN); }
        public MjDoubleVector gainprm() { return actuatorGainprm().slice(MjConstants.mjNGAIN * id, MjConstants.mjNGAIN); }
        public MjDoubleVector biasprm() { return actuatorBiasprm().slice(MjConstants.mjNBIAS * id, MjConstants.mjNBIAS); }
        public MjDoubleVector ctrlrange() { return actuatorCtrlrange().slice(2 * id, 2); }
        public MjDoubleVector forcerange() { return actuatorForcerange().slice(2 * id, 2); }
        public MjDoubleVector gear() { return actuatorGear().slice(6 * id, 6); }
        public MjDoubleVector lengthrange() { return actuatorLengthrange().slice(2 * id, 2); }
        public double delay() { return actuatorDelay().get(id); }
    }

    public final class SensorView {
        private final int id;
        private SensorView(int id) { this.id = id; }
        public int id() { checkOpen(); return id; }
        public String name() { checkOpen(); String n = id2name(MjObjType.SENSOR, id); return n != null ? n : ""; }
        public int type() { return sensorType().get(id); }
        public int datatype() { return sensorDatatype().get(id); }
        public int needstage() { return sensorNeedstage().get(id); }
        public int objtype() { return sensorObjtype().get(id); }
        public int objid() { return sensorObjid().get(id); }
        public int dim() { return sensorDim().get(id); }
        public int adr() { return sensorAdr().get(id); }
        public double cutoff() { return sensorCutoff().get(id); }
        public double noise() { return sensorNoise().get(id); }
        public double delay() { return sensorDelay().get(id); }
        public MjDoubleVector interval() { return sensorInterval().slice(2 * id, 2); }
    }

    public final class TexView {
        private final int id;
        private TexView(int id) { this.id = id; }
        public int id() { checkOpen(); return id; }
        public String name() { checkOpen(); String n = id2name(MjObjType.TEXTURE, id); return n != null ? n : ""; }
        public int type() { return texType().get(id); }
        public int colorspace() { return texColorspace().get(id); }
        public int height() { return texHeight().get(id); }
        public int width() { return texWidth().get(id); }
        public int nchannel() { return texNchannel().get(id); }
        public long adr() { return texAdr().get(id); }
        public MjByteVector data() {
            int offset = (int) adr();
            int len = height() * width() * nchannel();
            return texData().slice(offset, len);
        }
    }
}
