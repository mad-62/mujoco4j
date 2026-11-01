package org.mujoco4j;

import org.mujoco4j.internal.panama.mjContact_;
import org.mujoco4j.internal.panama.mjData_;

import java.lang.foreign.MemorySegment;
import java.util.AbstractList;
import java.util.ArrayList;
import java.util.List;

/**
 * Dynamic live view over {@code d->contact} and {@code d->ncon}.
 * Because {@code d->contact} resides in {@code d->arena} and is reallocated on every simulation step,
 * this view resolves {@code d->ncon} and {@code d->contact} dynamically on every access.
 */
public final class MjContactList extends AbstractList<MjContact> {

    private final MjData data;

    MjContactList(MjData data) {
        this.data = data;
    }

    @Override
    public int size() {
        data.checkOpen();
        return mjData_.ncon(data.segment());
    }

    @Override
    public boolean isEmpty() {
        return size() == 0;
    }

    @Override
    public MjContact get(int index) {
        data.checkOpen();
        int ncon = mjData_.ncon(data.segment());
        if (index < 0 || index >= ncon) {
            throw new IndexOutOfBoundsException("Contact index " + index + " out of bounds for ncon " + ncon);
        }
        MemorySegment contactPtr = mjData_.contact(data.segment());
        if (MemorySegment.NULL.equals(contactPtr)) {
            throw new IllegalStateException("d->contact is NULL while ncon=" + ncon);
        }
        MemorySegment arraySeg = contactPtr.reinterpret((long) ncon * mjContact_.sizeof(), data.arena(), null);
        MemorySegment elemSeg = mjContact_.asSlice(arraySeg, index);
        return new MjContact(elemSeg, data::isOpen);
    }

    public List<MjContact> copyAll() {
        int n = size();
        List<MjContact> copies = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            copies.add(get(i).copy());
        }
        return copies;
    }
}
