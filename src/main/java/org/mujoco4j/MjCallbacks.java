package org.mujoco4j;

import org.mujoco4j.internal.panama.mjfGeneric;
import org.mujoco4j.internal.panama.mjfSensor;
import org.mujoco4j.internal.panama.mjfTime;
import org.mujoco4j.internal.panama.mju_user_warning;
import org.mujoco4j.internal.panama.mujoco_h;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.ref.WeakReference;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.function.DoubleSupplier;

/**
 * Manages global MuJoCo pipeline and warning callbacks ({@code mjcb_time}, {@code mjcb_control},
 * {@code mjcb_sensor}, {@code mju_user_warning}) using Java 25 FFM upcall stubs.
 *
 * <p>Exceptions thrown inside Java callbacks are trapped before crossing the native C frame boundary
 * and rethrown cleanly on the Java side via {@link #rethrowIfPending()}.
 */
public final class MjCallbacks {

    @FunctionalInterface
    public interface SensorCallback {
        void accept(MjModel model, MjData data, MjStage stage);
    }

    private static final ConcurrentHashMap<Long, WeakReference<MjModel>> MODEL_REGISTRY = new ConcurrentHashMap<>();
    private static final ConcurrentHashMap<Long, WeakReference<MjData>> DATA_REGISTRY = new ConcurrentHashMap<>();
    private static final ThreadLocal<Throwable> PENDING_EXCEPTION = new ThreadLocal<>();

    private static MemorySegment defaultTimeStub;
    private static MemorySegment userTimeStub;
    private static MemorySegment controlStub;
    private static MemorySegment sensorStub;
    private static MemorySegment warningStub;

    private static volatile DoubleSupplier currentTimeCallback;
    private static volatile BiConsumer<MjModel, MjData> currentControlCallback;
    private static volatile SensorCallback currentSensorCallback;
    private static volatile Consumer<String> currentWarningCallback;

    private MjCallbacks() {
    }

    static void registerModel(MjModel model) {
        MODEL_REGISTRY.put(model.address(), new WeakReference<>(model));
    }

    static void unregisterModel(long address) {
        MODEL_REGISTRY.remove(address);
    }

    static void unregisterModel(MjModel model) {
        if (model != null) {
            MODEL_REGISTRY.remove(model.segmentUnchecked().address());
        }
    }

    static void registerData(MjData data) {
        DATA_REGISTRY.put(data.address(), new WeakReference<>(data));
    }

    static void unregisterData(long address) {
        DATA_REGISTRY.remove(address);
    }

    static void unregisterData(MjData data) {
        if (data != null) {
            DATA_REGISTRY.remove(data.segmentUnchecked().address());
        }
    }

    public static synchronized void clearAll() {
        setTimeCallback(null);
        setControlCallback(null);
        setSensorCallback(null);
        setWarningCallback(null);
        clearPendingException();
    }

    static synchronized void installDefaultTimerIfNeeded() {
        if (defaultTimeStub == null) {
            defaultTimeStub = mjfTime.allocate(() -> System.nanoTime() * 1e-6, Arena.global());
            userTimeStub = mjfTime.allocate(() -> {
                DoubleSupplier cb = currentTimeCallback;
                if (cb == null) {
                    return System.nanoTime() * 1e-6;
                }
                try {
                    return cb.getAsDouble();
                } catch (Throwable t) {
                    PENDING_EXCEPTION.set(t);
                    return 0.0;
                }
            }, Arena.global());
            controlStub = mjfGeneric.allocate((mPtr, dPtr) -> {
                BiConsumer<MjModel, MjData> cb = currentControlCallback;
                if (cb == null) {
                    return;
                }
                try {
                    MjModel m = resolveModel(mPtr);
                    MjData d = resolveData(dPtr, m);
                    cb.accept(m, d);
                } catch (Throwable t) {
                    PENDING_EXCEPTION.set(t);
                }
            }, Arena.global());
            sensorStub = mjfSensor.allocate((mPtr, dPtr, stage) -> {
                SensorCallback cb = currentSensorCallback;
                if (cb == null) {
                    return;
                }
                try {
                    MjModel m = resolveModel(mPtr);
                    MjData d = resolveData(dPtr, m);
                    cb.accept(m, d, MjStage.fromInt(stage));
                } catch (Throwable t) {
                    PENDING_EXCEPTION.set(t);
                }
            }, Arena.global());
            warningStub = mju_user_warning.allocate(msgPtr -> {
                Consumer<String> cb = currentWarningCallback;
                if (cb == null) {
                    return;
                }
                try {
                    String msg = MemorySegment.NULL.equals(msgPtr) ? "" : msgPtr.reinterpret(4096).getString(0);
                    cb.accept(msg);
                } catch (Throwable t) {
                    PENDING_EXCEPTION.set(t);
                }
            }, Arena.global());

            mujoco_h.mjcb_time(defaultTimeStub);
        }
    }

    public static synchronized void setTimeCallback(DoubleSupplier callback) {
        MjNativeLoader.ensureLoaded();
        currentTimeCallback = callback;
        mujoco_h.mjcb_time(callback != null ? userTimeStub : defaultTimeStub);
    }

    public static DoubleSupplier getTimeCallback() {
        MjNativeLoader.ensureLoaded();
        return currentTimeCallback;
    }

    public static synchronized void setControlCallback(BiConsumer<MjModel, MjData> callback) {
        MjNativeLoader.ensureLoaded();
        currentControlCallback = callback;
        mujoco_h.mjcb_control(callback != null ? controlStub : MemorySegment.NULL);
    }

    public static BiConsumer<MjModel, MjData> getControlCallback() {
        MjNativeLoader.ensureLoaded();
        return currentControlCallback;
    }

    public static synchronized void setSensorCallback(SensorCallback callback) {
        MjNativeLoader.ensureLoaded();
        currentSensorCallback = callback;
        mujoco_h.mjcb_sensor(callback != null ? sensorStub : MemorySegment.NULL);
    }

    public static SensorCallback getSensorCallback() {
        MjNativeLoader.ensureLoaded();
        return currentSensorCallback;
    }

    public static synchronized void setWarningCallback(Consumer<String> callback) {
        MjNativeLoader.ensureLoaded();
        currentWarningCallback = callback;
        mujoco_h.mju_user_warning(callback != null ? warningStub : MemorySegment.NULL);
    }

    public static Consumer<String> getWarningCallback() {
        MjNativeLoader.ensureLoaded();
        return currentWarningCallback;
    }

    public static void clearPendingException() {
        PENDING_EXCEPTION.remove();
    }

    public static void rethrowIfPending() {
        Throwable t = PENDING_EXCEPTION.get();
        if (t != null) {
            PENDING_EXCEPTION.remove();
            if (t instanceof RuntimeException re) {
                throw re;
            } else if (t instanceof Error err) {
                throw err;
            } else {
                throw new MjException("Uncaught exception in MuJoCo callback", t);
            }
        }
    }

    private static MjModel resolveModel(MemorySegment mPtr) {
        WeakReference<MjModel> ref = MODEL_REGISTRY.get(mPtr.address());
        return ref != null ? ref.get() : null;
    }

    private static MjData resolveData(MemorySegment dPtr, MjModel m) {
        WeakReference<MjData> ref = DATA_REGISTRY.get(dPtr.address());
        return ref != null ? ref.get() : null;
    }
}
