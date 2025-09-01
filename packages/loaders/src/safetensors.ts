/** Supported data types for loading from Safetensors format. */
export type DType =
  | "F16" // Float16Array
  | "F32" // Float32Array
  | "F64" // Float64Array
  | "I8" // Int8Array
  | "I16" // Int16Array
  | "I32" // Int32Array
  | "I64" // BigInt64Array
  | "U8" // Uint8Array
  | "U16" // Uint16Array
  | "U32" // Uint32Array
  | "U64" // BigUint64Array
  | "BOOL"; // Represented as Uint8Array

export type Tensor = {
  dtype: DType;
  shape: number[];
  data: TensorData;
};

export type TensorData =
  | Float16Array
  | Float32Array
  | Float64Array
  | Int8Array
  | Int16Array
  | Int32Array
  | BigInt64Array
  | Uint8Array
  | Uint16Array
  | Uint32Array
  | BigUint64Array;

export type File = {
  tensors: { [key: string]: Tensor };
  metadata?: Record<string, string>;
  totalSize: number;
};

/** Load data from a safetensors file. */
export function parse(data: Uint8Array | ArrayBuffer): File {
  let buffer: ArrayBuffer;
  let ptr = 0;
  const len = data.byteLength;
  if (data instanceof ArrayBuffer) {
    buffer = data;
  } else {
    buffer = data.buffer;
    ptr = data.byteOffset;
  }

  if (len < 8) throw new Error("Data too short to be a valid safetensors file");
  const view = new DataView(buffer, ptr, 8);

  const headerSize = view.getBigUint64(0, true);
  if (headerSize > BigInt(len - 8)) throw new Error("Invalid header size");

  let header: any;
  try {
    header = JSON.parse(
      new TextDecoder().decode(
        new Uint8Array(buffer, ptr + 8, Number(headerSize)),
      ),
    );
  } catch (error) {
    throw new Error(`Failed to parse safetensors header as JSON: ${error}`);
  }

  const file: File = { tensors: {}, totalSize: len };
  for (const [key, value] of Object.entries(header)) {
    if (key === "__metadata__") {
      file.metadata = value as Record<string, string>;
      continue;
    }
    const { dtype, shape, data_offsets } = value as {
      dtype: DType;
      shape: number[];
      data_offsets: [number, number];
    };
    const byteOffset = ptr + Number(headerSize) + 8 + data_offsets[0];
    const byteLength = data_offsets[1] - data_offsets[0];
    let data: TensorData;
    switch (dtype) {
      case "F16":
        data = new Float16Array(buffer, byteOffset, byteLength / 2);
        break;
      case "F32":
        data = new Float32Array(buffer, byteOffset, byteLength / 4);
        break;
      case "F64":
        data = new Float64Array(buffer, byteOffset, byteLength / 8);
        break;
      case "I8":
        data = new Int8Array(buffer, byteOffset, byteLength);
        break;
      case "I16":
        data = new Int16Array(buffer, byteOffset, byteLength / 2);
        break;
      case "I32":
        data = new Int32Array(buffer, byteOffset, byteLength / 4);
        break;
      case "I64":
        data = new BigInt64Array(buffer, byteOffset, byteLength / 8);
        break;
      case "U8":
        data = new Uint8Array(buffer, byteOffset, byteLength);
        break;
      case "U16":
        data = new Uint16Array(buffer, byteOffset, byteLength / 2);
        break;
      case "U32":
        data = new Uint32Array(buffer, byteOffset, byteLength / 4);
        break;
      case "U64":
        data = new BigUint64Array(buffer, byteOffset, byteLength / 8);
        break;
      case "BOOL":
        data = new Uint8Array(buffer, byteOffset, byteLength);
        break;
      default:
        throw new Error(`Unsupported dtype: ${dtype}`);
    }
    file.tensors[key] = { dtype, shape, data };
  }
  return file;
}
