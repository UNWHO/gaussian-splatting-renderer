import { render } from "../../wasm/pkg/gs";
import { loadFile } from "./ply";
import preprocessShader from "../../wasm/src/shader/preprocess.wgsl?raw";
import {
  getSizeAndAlignmentOfUnsizedArrayElement,
  makeShaderDataDefinitions,
  makeStructuredView,
} from "webgpu-utils";
import { loadCamera, preprocess } from "./util";

const file = document.getElementsByTagName("input")[0];

file.addEventListener("change", async (e) => {
  const input = e.target as HTMLInputElement;
  const file = input.files!.item(0)!;

  const gaussians = await loadFile(file);

  const processedGaussian = gaussians.map(preprocess);

  const defs = makeShaderDataDefinitions(preprocessShader);
  const { size } = getSizeAndAlignmentOfUnsizedArrayElement(
    defs.storages.gaussians
  );
  const gaussianStructure = makeStructuredView(
    defs.storages.gaussians,
    new ArrayBuffer(size * gaussians.length)
  );

  gaussianStructure.set(processedGaussian);

  const cameras = loadCamera();

  // for (const camera of cameras) {
  render(
    new Float32Array(gaussianStructure.arrayBuffer),
    BigInt(gaussians.length),
    new Float32Array(cameras[0].cameraParam),
    new Uint32Array(cameras[0].sizeParam)
  );
  // }
});
