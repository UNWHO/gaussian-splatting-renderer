// import ply_file from "../public/point_cloud.ply?url";

const properties = [
  "x",
  "y",
  "z",
  "nx",
  "ny",
  "nz",
  "f_dc_0",
  "f_dc_1",
  "f_dc_2",
  "f_rest_0",
  "f_rest_1",
  "f_rest_2",
  "f_rest_3",
  "f_rest_4",
  "f_rest_5",
  "f_rest_6",
  "f_rest_7",
  "f_rest_8",
  "f_rest_9",
  "f_rest_10",
  "f_rest_11",
  "f_rest_12",
  "f_rest_13",
  "f_rest_14",
  "f_rest_15",
  "f_rest_16",
  "f_rest_17",
  "f_rest_18",
  "f_rest_19",
  "f_rest_20",
  "f_rest_21",
  "f_rest_22",
  "f_rest_23",
  "f_rest_24",
  "f_rest_25",
  "f_rest_26",
  "f_rest_27",
  "f_rest_28",
  "f_rest_29",
  "f_rest_30",
  "f_rest_31",
  "f_rest_32",
  "f_rest_33",
  "f_rest_34",
  "f_rest_35",
  "f_rest_36",
  "f_rest_37",
  "f_rest_38",
  "f_rest_39",
  "f_rest_40",
  "f_rest_41",
  "f_rest_42",
  "f_rest_43",
  "f_rest_44",
  "opacity",
  "scale_0",
  "scale_1",
  "scale_2",
  "rot_0",
  "rot_1",
  "rot_2",
  "rot_3",
] as const;
type Property = (typeof properties)[number];

enum DataType {
  CHAR = "char",
  UCHAR = "uchar",
  SHORT = "short",
  USHORT = "ushort",
  INT = "int",
  UINT = "uint",
  FLOAT = "float",
  DOUBLE = "double",

  INT8 = "int8",
  UINT8 = "uint8",
  INT16 = "int16",
  UINT16 = "uint16",
  INT32 = "int32",
  UINT32 = "uint32",
  FLOAT32 = "float32",
  FLOAT64 = "float64",
}

export type Gaussian = Record<Property, number>;

export const loadFile = async (file: File): Promise<Gaussian[]> => {
  const arrayBuffer = await file.arrayBuffer();

  // parse header
  const typedArray = new Uint8Array(arrayBuffer);

  let start = 0;
  let end = 0;
  const decoder = new TextDecoder();

  const headers: string[] = [];

  while (true) {
    if (typedArray[end] == 10) {
      const str = decoder.decode(typedArray.subarray(start, end));
      headers.push(str);

      if (str === "end_header") break;

      start = end + 1;
    }
    end++;
  }

  // delete useless header
  headers.shift();
  headers.pop();

  const isLittleEndian = headers.shift()?.split("_")[1] === "little";
  const numVertex = Number(headers.shift()?.split(" ")[2]);
  console.log("Number of gaussian: ", numVertex);

  const properties = headers.map((header) => {
    const [_, type, name] = header.split(" ");
    return { type: type as DataType, name: name as Property };
  });

  const dataview = new DataView(arrayBuffer);

  let index = end + 1;
  const gaussians: Gaussian[] = [];

  while (index < arrayBuffer.byteLength) {
    const vertex = {} as Gaussian;

    properties.forEach((property) => {
      const { name, type } = property;

      switch (type) {
        case DataType.CHAR:
        case DataType.INT8:
          vertex[name] = dataview.getInt8(index);
          index += 1;
          break;
        case DataType.UCHAR:
        case DataType.UINT8:
          vertex[name] = dataview.getUint8(index);
          index += 1;
          break;
        case DataType.SHORT:
        case DataType.INT16:
          vertex[name] = dataview.getInt16(index, isLittleEndian);
          index += 2;
          break;
        case DataType.USHORT:
        case DataType.UINT16:
          vertex[name] = dataview.getUint16(index, isLittleEndian);
          index += 2;
          break;
        case DataType.INT:
        case DataType.INT32:
          vertex[name] = dataview.getInt32(index, isLittleEndian);
          index += 4;
          break;
        case DataType.UINT:
        case DataType.UINT32:
          vertex[name] = dataview.getUint32(index, isLittleEndian);
          index += 4;
          break;
        case DataType.FLOAT:
        case DataType.FLOAT32:
          vertex[name] = dataview.getFloat32(index, isLittleEndian);
          index += 4;
          break;
        case DataType.DOUBLE:
        case DataType.FLOAT64:
          vertex[name] = dataview.getFloat64(index, isLittleEndian);
          index += 8;
      }
    });

    gaussians.push(vertex);
    // break;
  }

  // return gaussians.map((gaussian) => ({
  //   ...gaussians[0],
  //   opacity: -10,
  //   x: 0.05,
  //   y: 0.05,
  //   z: 1.0,
  //   scale_0: -2,
  //   scale_1: -2,
  //   scale_2: -2,
  //   f_dc_0: 0.0,
  //   f_dc_1: 1.0,
  //   f_dc_2: 0.0,
  // }));

  // return [
  //   {
  //     ...gaussians[0],
  //     opacity: 10,
  //     x: 0.05,
  //     y: 0.05,
  //     z: 1.0,
  //     scale_0: -2,
  //     scale_1: -2,
  //     scale_2: -2,
  //     f_dc_0: 0.0,
  //     f_dc_1: 1.0,
  //     f_dc_2: 0.0,
  //   },
  //   {
  //     ...gaussians[1],
  //     opacity: 10,
  //     x: 0,
  //     y: -0.0,
  //     z: 0.5,
  //     scale_0: -1,
  //     scale_1: -3,
  //     scale_2: -3,
  //     f_dc_0: 1.0,
  //     f_dc_1: 0.0,
  //     f_dc_2: 0.0,
  //   },
  // ];

  return gaussians;
};
