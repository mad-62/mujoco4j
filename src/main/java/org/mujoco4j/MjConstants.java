package org.mujoco4j;

import org.mujoco4j.internal.panama.mujoco_h;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Global MuJoCo constants and string lookup tables backed directly by the native library.
 */
public final class MjConstants {

    public static final int mjVERSION_HEADER = mujoco_h.mjVERSION_HEADER();
    public static final int VERSION_HEADER = mjVERSION_HEADER;

    public static final double mjMINVAL = mujoco_h.mjMINVAL();
    public static final double mjMAXVAL = mujoco_h.mjMAXVAL();
    public static final double mjPI = mujoco_h.mjPI();
    public static final double mjMAXIMP = mujoco_h.mjMAXIMP();
    public static final double mjMINIMP = mujoco_h.mjMINIMP();

    public static final double MINVAL = mjMINVAL;
    public static final double MAXVAL = mjMAXVAL;
    public static final double PI = mjPI;
    public static final double MAXIMP = mjMAXIMP;
    public static final double MINIMP = mjMINIMP;

    public static final int mjNREF = mujoco_h.mjNREF();
    public static final int mjNIMP = mujoco_h.mjNIMP();
    public static final int mjNPOLY = mujoco_h.mjNPOLY();
    public static final int mjNDYN = mujoco_h.mjNDYN();
    public static final int mjNGAIN = mujoco_h.mjNGAIN();
    public static final int mjNBIAS = mujoco_h.mjNBIAS();
    public static final int mjNFLUID = mujoco_h.mjNFLUID();
    public static final int mjNEQDATA = mujoco_h.mjNEQDATA();
    public static final int mjNSENS = mujoco_h.mjNSENS();
    public static final int mjNGROUP = mujoco_h.mjNGROUP();
    public static final int mjNSOLVER = mujoco_h.mjNSOLVER();
    public static final int mjNISLAND = mujoco_h.mjNISLAND();
    public static final int mjNDISABLE = mujoco_h.mjNDISABLE();
    public static final int mjNENABLE = mujoco_h.mjNENABLE();
    public static final int mjNTIMER = mujoco_h.mjNTIMER();
    public static final int mjNLABEL = mujoco_h.mjNLABEL();
    public static final int mjNFRAME = mujoco_h.mjNFRAME();
    public static final int mjNVISFLAG = mujoco_h.mjNVISFLAG();
    public static final int mjNRNDFLAG = mujoco_h.mjNRNDFLAG();
    public static final int mjNWARNING = mujoco_h.mjNWARNING();
    public static final int mjNSTATE = mujoco_h.mjNSTATE();

    public static final int NREF = mjNREF;
    public static final int NIMP = mjNIMP;
    public static final int NPOLY = mjNPOLY;
    public static final int NDYN = mjNDYN;
    public static final int NGAIN = mjNGAIN;
    public static final int NBIAS = mjNBIAS;
    public static final int NFLUID = mjNFLUID;
    public static final int NEQDATA = mjNEQDATA;
    public static final int NSENS = mjNSENS;
    public static final int NGROUP = mjNGROUP;
    public static final int NSOLVER = mjNSOLVER;
    public static final int NISLAND = mjNISLAND;
    public static final int NDISABLE = mjNDISABLE;
    public static final int NENABLE = mjNENABLE;
    public static final int NTIMER = mjNTIMER;
    public static final int NLABEL = mjNLABEL;
    public static final int NFRAME = mjNFRAME;
    public static final int NVISFLAG = mjNVISFLAG;
    public static final int NRNDFLAG = mjNRNDFLAG;
    public static final int NWARNING = mjNWARNING;
    public static final int NSTATE = mjNSTATE;

    public static final int mjJNT_FREE = mujoco_h.mjJNT_FREE();
    public static final int mjJNT_BALL = mujoco_h.mjJNT_BALL();
    public static final int mjJNT_SLIDE = mujoco_h.mjJNT_SLIDE();
    public static final int mjJNT_HINGE = mujoco_h.mjJNT_HINGE();

    private MjConstants() {
    }

    public static List<String> disableStrings() {
        MjNativeLoader.ensureLoaded();
        return readStringArray(mujoco_h.mjDISABLESTRING(), NDISABLE);
    }

    public static List<String> enableStrings() {
        MjNativeLoader.ensureLoaded();
        return readStringArray(mujoco_h.mjENABLESTRING(), NENABLE);
    }

    public static List<String> timerStrings() {
        MjNativeLoader.ensureLoaded();
        return readStringArray(mujoco_h.mjTIMERSTRING(), NTIMER);
    }

    public static List<String> labelStrings() {
        MjNativeLoader.ensureLoaded();
        return readStringArray(mujoco_h.mjLABELSTRING(), NLABEL);
    }

    public static List<String> frameStrings() {
        MjNativeLoader.ensureLoaded();
        return readStringArray(mujoco_h.mjFRAMESTRING(), NFRAME);
    }

    public static List<String[]> visStrings() {
        MjNativeLoader.ensureLoaded();
        return readStringTriplets(mujoco_h.mjVISSTRING(), NVISFLAG);
    }

    public static List<String[]> rndStrings() {
        MjNativeLoader.ensureLoaded();
        return readStringTriplets(mujoco_h.mjRNDSTRING(), NRNDFLAG);
    }

    private static List<String> readStringArray(MemorySegment base, int count) {
        List<String> list = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            MemorySegment ptr = base.getAtIndex(ValueLayout.ADDRESS, i);
            list.add(ptr.reinterpret(256).getString(0));
        }
        return Collections.unmodifiableList(list);
    }

    private static List<String[]> readStringTriplets(MemorySegment base, int rows) {
        List<String[]> list = new ArrayList<>(rows);
        for (int r = 0; r < rows; r++) {
            String[] row = new String[3];
            for (int c = 0; c < 3; c++) {
                MemorySegment ptr = base.getAtIndex(ValueLayout.ADDRESS, (long) r * 3 + c);
                row[c] = ptr.reinterpret(256).getString(0);
            }
            list.add(row);
        }
        return Collections.unmodifiableList(list);
    }
}
