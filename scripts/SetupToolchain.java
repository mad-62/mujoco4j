///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25+

import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

/**
 * Self-bootstrapping toolchain & Layer 1 codegen script for mujoco4j.
 * Can be run via:
 *   java scripts/SetupToolchain.java
 * or:
 *   jbang scripts/SetupToolchain.java
 */
public class SetupToolchain {

    public static final String MUJOCO_VERSION = "3.17.0";

    public static void main(String[] args) throws Exception {
        Path root = Path.of("").toAbsolutePath();
        Path mujocoDir = root.resolve(".mujoco");
        Path includeDir = mujocoDir.resolve("include");
        Path mujocoHeader = includeDir.resolve("mujoco").resolve("mujoco.h");
        Path shimIncludeDir = mujocoDir.resolve("shim-include");
        Path jextractBin = root.resolve(".tools").resolve("jextract-22").resolve("bin")
                .resolve(isWindows() ? "jextract.bat" : "jextract");

        if (!Files.exists(mujocoHeader)) {
            throw new IllegalStateException("Missing MuJoCo header at: " + mujocoHeader);
        }
        if (!Files.exists(jextractBin)) {
            throw new IllegalStateException("Missing jextract binary at: " + jextractBin);
        }

        // 0. Ensure GLFW 3.4 native library (glfw3.dll) is present in .mujoco/bin
        Path glfwDll = mujocoDir.resolve("bin").resolve("glfw3.dll");
        if (isWindows() && !Files.exists(glfwDll)) {
            System.out.println("Downloading GLFW 3.4 Windows x64 binary...");
            URI glfwUri = URI.create("https://github.com/glfw/glfw/releases/download/3.4/glfw-3.4.bin.WIN64.zip");
            try (InputStream in = glfwUri.toURL().openStream();
                 ZipInputStream zis = new ZipInputStream(in)) {
                ZipEntry entry;
                while ((entry = zis.getNextEntry()) != null) {
                    if (entry.getName().endsWith("lib-vc2022/glfw3.dll")) {
                        Files.createDirectories(glfwDll.getParent());
                        Files.copy(zis, glfwDll, StandardCopyOption.REPLACE_EXISTING);
                        System.out.println("Extracted GLFW 3.4 to: " + glfwDll);
                        break;
                    }
                }
            }
        }

        // 1. Create hermetic CRT shim headers (stdlib.h and math.h) for libclang on Windows
        Files.createDirectories(shimIncludeDir);
        Files.writeString(shimIncludeDir.resolve("stdlib.h"), """
                #ifndef _MUJOCO4J_SHIM_STDLIB_H_
                #define _MUJOCO4J_SHIM_STDLIB_H_
                #include <stddef.h>
                #endif
                """);
        Files.writeString(shimIncludeDir.resolve("math.h"), """
                #ifndef _MUJOCO4J_SHIM_MATH_H_
                #define _MUJOCO4J_SHIM_MATH_H_
                double fabs(double x);
                double log10(double x);
                #endif
                """);

        // 2. Dump all symbols discovered by jextract from mujoco.h
        Path rawIncludes = mujocoDir.resolve("jextract-all-includes.txt");
        runCommand(List.of(
                jextractBin.toString(),
                "-I", includeDir.toString(),
                "-I", shimIncludeDir.toString(),
                "--dump-includes", rawIncludes.toString(),
                mujocoHeader.toString()
        ));

        // 3. Filter includes to keep only MuJoCo symbols (exclude shim math.h / stdlib.h symbols)
        List<String> filteredLines = new ArrayList<>();
        for (String line : Files.readAllLines(rawIncludes)) {
            String trimmed = line.trim();
            if (trimmed.isEmpty() || trimmed.startsWith("#")) {
                continue;
            }
            if (trimmed.contains("fabs") || trimmed.contains("log10") || trimmed.contains("_MUJOCO4J_SHIM")) {
                continue;
            }
            filteredLines.add(trimmed);
        }
        Path filteredIncludes = mujocoDir.resolve("jextract-mujoco-includes.txt");
        Files.write(filteredIncludes, filteredLines);
        System.out.println("Filtered MuJoCo symbols count: " + filteredLines.size());

        // 4. Clean existing generated Layer 1 directory and run jextract
        Path outputJavaDir = root.resolve("src").resolve("main").resolve("java");
        Path panamaDir = outputJavaDir.resolve("org").resolve("mujoco4j").resolve("internal").resolve("panama");
        if (Files.exists(panamaDir)) {
            try (var walk = Files.walk(panamaDir)) {
                walk.sorted((a, b) -> b.compareTo(a)).forEach(p -> {
                    try {
                        Files.deleteIfExists(p);
                    } catch (IOException ignored) {
                    }
                });
            }
        }
        Files.createDirectories(outputJavaDir);

        runCommand(List.of(
                jextractBin.toString(),
                "-I", includeDir.toString(),
                "-I", shimIncludeDir.toString(),
                "--output", outputJavaDir.toString(),
                "--target-package", "org.mujoco4j.internal.panama",
                "--header-class-name", "mujoco_h",
                "@" + filteredIncludes.toString(),
                mujocoHeader.toString()
        ));

        System.out.println("Successfully generated Layer 1 FFM bindings in: " + panamaDir);
    }

    private static void runCommand(List<String> cmd) throws IOException, InterruptedException {
        System.out.println("Running: " + String.join(" ", cmd));
        ProcessBuilder pb = new ProcessBuilder(cmd);
        pb.inheritIO();
        Process proc = pb.start();
        int exit = proc.waitFor();
        if (exit != 0) {
            throw new RuntimeException("Command failed with exit code " + exit + ": " + cmd);
        }
    }

    private static boolean isWindows() {
        return System.getProperty("os.name", "").toLowerCase().contains("win");
    }
}
