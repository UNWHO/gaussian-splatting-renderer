import { Gaussian } from "./ply";
import cameraJson from "../public/cameras.json";
import { Matrix3, Matrix4, Object3D, Quaternion, Vector3 } from "three";
import { mat3, mat4, vec3 } from "wgpu-matrix";

const normalize = (...elements: number[]): number[] => {
  const magnitude = elements.map((e) => e * e).reduce((acc, e) => acc + e, 0);
  return elements.map((e) => e / Math.sqrt(magnitude));
};

const sigmoid = (x: number): number => {
  return 1 / (Math.exp(-x) + 1);
};

export const preprocess = (gaussian: Gaussian) => {
  return {
    mean: [gaussian.x, gaussian.y, gaussian.z],
    norm: [gaussian.nx, gaussian.ny, gaussian.nz],
    sh: [
      [gaussian.f_dc_0, gaussian.f_dc_1, gaussian.f_dc_2],
      [gaussian.f_rest_0, gaussian.f_rest_1, gaussian.f_rest_2],
      [gaussian.f_rest_3, gaussian.f_rest_4, gaussian.f_rest_5],
      [gaussian.f_rest_6, gaussian.f_rest_7, gaussian.f_rest_8],
      [gaussian.f_rest_9, gaussian.f_rest_10, gaussian.f_rest_11],
      [gaussian.f_rest_12, gaussian.f_rest_13, gaussian.f_rest_14],
      [gaussian.f_rest_15, gaussian.f_rest_16, gaussian.f_rest_17],
      [gaussian.f_rest_18, gaussian.f_rest_19, gaussian.f_rest_20],
      [gaussian.f_rest_21, gaussian.f_rest_22, gaussian.f_rest_23],
      [gaussian.f_rest_24, gaussian.f_rest_25, gaussian.f_rest_26],
      [gaussian.f_rest_27, gaussian.f_rest_28, gaussian.f_rest_29],
      [gaussian.f_rest_30, gaussian.f_rest_31, gaussian.f_rest_32],
      [gaussian.f_rest_33, gaussian.f_rest_34, gaussian.f_rest_35],
      [gaussian.f_rest_36, gaussian.f_rest_37, gaussian.f_rest_38],
      [gaussian.f_rest_39, gaussian.f_rest_40, gaussian.f_rest_41],
      [gaussian.f_rest_42, gaussian.f_rest_43, gaussian.f_rest_44],
    ],
    scale: [gaussian.scale_0, gaussian.scale_1, gaussian.scale_2].map(Math.exp),
    opacity: sigmoid(gaussian.opacity),
    rotation: normalize(
      gaussian.rot_0,
      gaussian.rot_1,
      gaussian.rot_2,
      gaussian.rot_3
    ),
  };
};

export const loadCamera = () => {
  return cameraJson.map((json) => {
    const obj = new Object3D();

    obj.translateX(json.position[0]);
    obj.translateY(json.position[1]);
    obj.translateZ(json.position[2]);

    let rotation = new Matrix4()
      .identity()
      .setFromMatrix3(
        new Matrix3().set(
          json.rotation[0][0],
          json.rotation[1][0],
          json.rotation[2][0],
          json.rotation[0][1],
          json.rotation[1][1],
          json.rotation[2][1],
          json.rotation[0][2],
          json.rotation[1][2],
          json.rotation[2][2]
        )
      );

    obj.setRotationFromQuaternion(
      new Quaternion().setFromRotationMatrix(rotation)
    );

    obj.updateMatrix();
    // obj.updateMatrixWorld()
    const dir = new Vector3();
    obj.getWorldDirection(dir);

    const R = mat3.create(...json.rotation.flat());
    const T = json.position;

    const V = mat4.fromMat3(R);
    // const t = vec3.mulScalar(T, -1);
    mat4.translate(V, T, V);

    // console.log(V);

    return {
      cameraParam: [
        ...obj.position.toArray(),
        ...dir.toArray().map((e, i) => e + json.position[i]),
        ...obj.up.multiplyScalar(-1).toArray(),
        json.fx,
        json.fy,
      ],
      sizeParam: [json.width, json.height],
    };
  });
};
