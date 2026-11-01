package org.mujoco4j;

/**
 * Unchecked exception thrown when a MuJoCo XML parse, model compile,
 * precondition check, or runtime operation fails.
 */
public class MjException extends RuntimeException {

    public MjException(String message) {
        super(message);
    }

    public MjException(String message, Throwable cause) {
        super(message, cause);
    }
}
