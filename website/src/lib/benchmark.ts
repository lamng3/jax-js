import type tf from "@tensorflow/tfjs";

export async function importTfjs(
  backend: "wasm" | "webgpu",
): Promise<typeof tf> {
  const tf = await import("@tensorflow/tfjs");
  if (backend === "wasm") {
    if (!isSecureContext || !crossOriginIsolated) {
      alert("tfjs-wasm requires a secure context and cross-origin isolation.");
      throw new Error("Insecure context for tfjs-wasm backend.");
    }
    const { setThreadsCount, setWasmPaths } = await import(
      "@tensorflow/tfjs-backend-wasm"
    );
    setThreadsCount(navigator.hardwareConcurrency);
    setWasmPaths(
      `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${tf.version.tfjs}/wasm-out/`,
    );
    await tf.setBackend("wasm");
  } else if (backend === "webgpu") {
    await import("@tensorflow/tfjs-backend-webgpu");
    await tf.setBackend("webgpu");
  } else {
    throw new Error(`Unsupported backend: ${backend}`);
  }
  return tf;
}
