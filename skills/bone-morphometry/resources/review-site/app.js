import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  resolveParameter,
  mountParameterCatalog,
  appendParameterDetails,
} from "./parameter-panel.js";
const catalogResponse = await fetch("/parameters.json");
if (!catalogResponse.ok) throw Error("Parameter definitions unavailable");
const catalog = await catalogResponse.json();
const $ = (id) => document.getElementById(id);
const response = await fetch("/assets/manifest.json");
if (!response.ok) throw Error("Could not load review manifest");
const data = await response.json();
for (const p of catalog.parameters)
  Object.assign(p, data.parameter_overrides?.[p.id] || {});
const alignedSource = () =>
  $("source").value === "tibia" || $("source").value.startsWith("iioc-");
const names = {
  femoral_length_provisional: "Femoral length",
  femoral_width_provisional: "Femoral width",
  lateral_tibial_height_V195: "Lateral tibial height at V195",
  tibial_growth_plate_junction_chord: "Junction chord (exploratory)",
  femoral_width_length_provisional: "Femoral width / length",
  patella_core_threshold_volume: "Patella mineralized volume",
  separate_proximal_patellar_ossification_volume:
    "Separate patellar ossification",
};
const viewport = $("viewport"),
  scene = new THREE.Scene();
scene.background = new THREE.Color("#101820");
const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 200);
camera.up.set(0, 0, -1);
const renderer = new THREE.WebGLRenderer({
  antialias: true,
  preserveDrawingBuffer: true,
});
renderer.localClippingEnabled = true;
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
viewport.append(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.minDistance = 0.3;
controls.maxDistance = 55;
scene.add(new THREE.HemisphereLight(0xffffff, 0x526475, 2));
const light = new THREE.DirectionalLight(0xffffff, 2);
scene.add(light);
const labelRay = new THREE.Raycaster();
const meshes = new Map(),
  rulers = new THREE.Group();
scene.add(rulers);
let selected = "femoral_length_provisional",
  labels = [],
  slicePlane,
  cutPlane = null,
  visibilityFilter = null,
  dirty = true;
function measurement(key) {
  if (key.startsWith("deck-"))
    return resolveParameter(
      catalog.parameters.find((p) => `deck-${p.id}` === key),
      data,
    );
  if (data.measurements[key]) return data.measurements[key];
  const s = data.structures.find((s) => key === "structure-" + s.label);
  return {
    label: s.label,
    value: s.volume_mm3,
    unit: "mm^3",
    definition:
      s.definition ||
      "Review this segmentation and anatomical assignment. Volume describes only mineralized voxels inside the acquired field; femoral, tibial and fibular shafts are cropped. It is not a whole-bone volume.",
  };
}
for (const [key, m] of Object.entries(data.measurements))
  if (key.startsWith("iioc_") || key.startsWith("tibial_trab_"))
    names[key] = m.title || key.replaceAll("_", " ");
for (const s of data.structures)
  names["structure-" + s.label] = "Structure: " + s.name.replaceAll("-", " ");
const stateKey = "microct-review:" + data.grid_id;
let notes = {};
try {
  notes = JSON.parse(localStorage.getItem(stateKey) || "{}");
} catch {}
function v(a) {
  return new THREE.Vector3(...a);
}
function world(xyz) {
  return data.affine
    .slice(0, 3)
    .map(
      (row, i) =>
        row[3] -
        data.center_world_mm[i] +
        row.slice(0, 3).reduce((sum, n, j) => sum + n * xyz[j], 0),
    );
}
function view(mode) {
  let target = new THREE.Vector3(0, 0, 0),
    direction = new THREE.Vector3(-1, -1, -0.45),
    up = new THREE.Vector3(0, 0, -1),
    distance = 16;
  if (mode !== "overview") {
    const tibia = mode.startsWith("tibia-");
    mode = mode.replace("tibia-", "");
    const f = tibia ? data.frames.tibia : data.frames.femur,
      b = f.basis_columns_xyz;
    target = v(world(f.origin_xyz));
    direction = new THREE.Vector3(
      ...b.map((row) => row[mode === "joint" ? 2 : 1]),
    ).multiplyScalar(
      mode === "posterior" || (tibia && mode === "joint") ? -1 : 1,
    );
    up = new THREE.Vector3(
      ...b.map((row) => row[mode === "joint" ? 1 : 2]),
    ).multiplyScalar(mode === "joint" ? 1 : -1);
    distance = 10;
  }
  camera.up.copy(up);
  controls.target.copy(target);
  camera.position.copy(target).addScaledVector(direction.normalize(), distance);
  controls.update();
}
function mark(point, text, color) {
  const sphere = new THREE.Mesh(
    new THREE.SphereGeometry(0.045, 12, 8),
    new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      depthTest: !$("xray").checked,
    }),
  );
  sphere.position.copy(v(point));
  sphere.renderOrder = 10;
  rulers.add(sphere);
  const element = document.createElement("div");
  element.className = "landmark";
  element.textContent = text;
  viewport.append(element);
  labels.push({ element, point: v(point) });
}
function clearRulers() {
  for (const item of rulers.children) {
    item.geometry?.dispose();
    item.material?.dispose();
  }
  rulers.clear();
  for (const l of labels) l.element.remove();
  labels = [];
}
function select(key) {
  selected = key;
  const m = measurement(key);
  document
    .querySelectorAll(".metric")
    .forEach((b) => b.classList.toggle("active", b.dataset.key === key));
  clearRulers();
  $("details").replaceChildren();
  const heading = document.createElement("h3");
  heading.textContent = m.title || names[key];
  $("parameter-picker").value = key.startsWith("deck-") ? key : "";
  $("details").append(heading);
  const definition = document.createElement("p");
  definition.textContent =
    m.definition ||
    "Mineralized label volume from the original grid; separate ossifications are not combined. All structure volumes are segmented voxel volumes.";
  if (key === "femoral_width_length_provisional" && !m.parameter)
    definition.textContent =
      "Width divided by length. Both measurements remain provisional.";
  $("details").append(definition);
  if (!m.parameter && Number.isFinite(m.value)) {
    const result = document.createElement("p");
    result.textContent = `${m.value.toFixed(4)} ${m.unit === "1" ? "(fraction)" : m.unit || ""} · ${m.status || "Review estimate"}`;
    $("details").append(result);
  }
  if (m.evidence_stage === "tibial-trabecular") {
    for (const file of [
      "REPORT.md",
      "qc-coronal-100.png",
      "qc-coronal-220.png",
      "qc-sagittal.png",
      "qc-sphere-maps.png",
      "results.json",
    ]) {
      const link = document.createElement("a");
      link.href = `/assets/evidence/tibial-trabecular/${file}`;
      link.target = "_blank";
      link.textContent = `Review ${file}`;
      link.style.display = "block";
      $("details").append(link);
    }
    const inspect = document.createElement("button");
    inspect.textContent = "Inspect tibial trabecular segmentation";
    inspect.onclick = () => {
      $("isolate").click();
      view("tibia-anterior");
      $("focus").click();
    };
    $("details").append(inspect);
    $("source").value = m.label === 18 ? "iioc-trab-bone" : "iioc-trab-tissue";
    $("axis").value = "1";
    configureSlice();
    $("slice-index").value = 190;
    updateSlice();
  }
  appendParameterDetails(
    m,
    data,
    () => {
      $("isolate").click();
      if (m.parameter.context === "tibia" || m.parameter.context === "iioc") {
        $("source").value =
          m.parameter.context === "iioc" ? "iioc-tissue" : "tibia";
        $("axis").value = "1";
        configureSlice();
        $("slice-index").value = 195;
        updateSlice();
      }
      $("focus").click();
      $("viewport").scrollIntoView({ behavior: "smooth", block: "center" });
    },
    select,
  );
  if (m.points_scene_mm) {
    const color = key.includes("chord") ? 0xf0ad4e : 0x62e3ef;
    const points = m.points_scene_mm.map(v),
      line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(points),
        new THREE.LineBasicMaterial({
          color,
          transparent: true,
          depthTest: !$("xray").checked,
        }),
      );
    line.renderOrder = 10;
    rulers.add(line);
    const endpointKey = m.parameter?.source_measurement || key;
    const endpoints = endpointKey.includes("length")
      ? ["A", "C"]
      : endpointKey.includes("width")
        ? ["L", "M"]
        : ["P1", "P2"];
    m.points_scene_mm.forEach((p, i) => mark(p, endpoints[i], color));
    const midpoint = points[0].clone().add(points[1]).multiplyScalar(0.5);
    mark(midpoint.toArray(), `${m.value.toFixed(3)} mm`, color);
    const small = document.createElement("p");
    small.textContent =
      "Ruler coordinates are recorded in the original image frame. Original structures use 2-voxel display sampling; IIOC masks use 1-voxel sampling. Hidden rulers are projected through bone when enabled.";
    $("details").append(small);
    m.points_xyz.forEach((p, i) => {
      const b = document.createElement("button");
      b.textContent = `Inspect ${endpoints[i]} slice`;
      b.onclick = () => {
        $("source").value = "filtered";
        $("axis").value = "0";
        configureSlice();
        $("slice-index").value = Math.round(p[2]);
        updateSlice();
      };
      $("details").append(b);
    });
  }
  if (m.points_aligned_uvw) {
    $("source").value = "iioc-tissue";
    $("axis").value = "1";
    configureSlice();
    $("slice-index").value = Math.round(m.points_aligned_uvw[0][1]);
    updateSlice();
  }
  if (key === "lateral_tibial_height_V195" || key.includes("chord")) {
    $("source").value = "tibia";
    $("axis").value = "1";
    configureSlice();
    $("slice-index").value = 195;
    updateSlice();
  }
  $("decision").value = notes[key]?.decision || "Not reviewed";
  $("note").value = notes[key]?.note || "";
}
for (const [key, m] of Object.entries(data.measurements)) {
  const b = document.createElement("button");
  b.className = "metric";
  b.dataset.key = key;
  const title = document.createElement("span");
  title.textContent = names[key] || key;
  const value = document.createElement("strong");
  value.textContent =
    m.value.toFixed(3) +
    (m.unit === "1"
      ? " (fraction)"
      : m.unit === "mm^3"
        ? " mm³"
        : ` ${m.unit}`);
  b.append(title, value);
  b.onclick = () => select(key);
  $(
    key.startsWith("tibial_trab_")
      ? "tibial-trabecular-metrics"
      : "measurements",
  ).append(b);
}
function saveNotes() {
  notes[selected] = {
    decision: $("decision").value,
    note: $("note").value,
    updated: new Date().toISOString(),
  };
  try {
    localStorage.setItem(stateKey, JSON.stringify(notes));
    $("saved").textContent =
      "Saved in this browser. Export to share or back up.";
  } catch {
    $("saved").textContent =
      "Browser storage unavailable. Export notes before leaving.";
  }
}
$("decision").onchange = saveNotes;
$("note").oninput = saveNotes;
function download(blob, name) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
$("export").onclick = () =>
  download(
    new Blob(
      [
        JSON.stringify(
          {
            sample: data.sample,
            grid_id: data.grid_id,
            exported: new Date().toISOString(),
            notes,
            measurements: data.measurements,
            parameter_definitions: catalog,
            slice_review: {
              source: $("source").value,
              axis: Number($("axis").value),
              index: Number($("slice-index").value),
              overlay: $("overlay").checked,
              cutaway: $("cutaway").checked,
              plane3d: $("plane3d").checked,
            },
            camera: {
              position: camera.position.toArray(),
              target: controls.target.toArray(),
              up: camera.up.toArray(),
            },
          },
          null,
          2,
        ),
      ],
      { type: "application/json" },
    ),
    "OA6-1RK-review.json",
  );
$("capture").onclick = () => {
  renderer.render(scene, camera);
  const canvas = document.createElement("canvas");
  canvas.width = renderer.domElement.width;
  canvas.height = renderer.domElement.height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(renderer.domElement, 0, 0);
  const scale = canvas.width / viewport.clientWidth;
  ctx.font = `${12 * scale}px sans-serif`;
  for (const { point, element } of labels) {
    const p = point.clone().project(camera);
    if (p.z < -1 || p.z > 1 || element.style.display === "none") continue;
    const x = ((p.x + 1) / 2) * canvas.width + 8 * scale,
      y = ((-p.y + 1) / 2) * canvas.height - 8 * scale;
    ctx.fillStyle = "#101820";
    ctx.fillRect(
      x - 3 * scale,
      y - 13 * scale,
      ctx.measureText(element.textContent).width + 6 * scale,
      18 * scale,
    );
    ctx.fillStyle = "white";
    ctx.fillText(element.textContent, x, y);
  }
  canvas.toBlob((b) => download(b, "OA6-1RK-3D.png"));
};
function visibility(id, visible) {
  meshes.get(id).visible = visible;
  $("structure-" + id).checked = visible;
}
$("all").onclick = () => {
  visibilityFilter = null;
  meshes.forEach((_, id) => visibility(id, true));
  dirty = true;
};
$("isolate").onclick = () => {
  const m = measurement(selected);
  const ids =
    m.labels ||
    (m.label ? [m.label] : selected.startsWith("femoral") ? [1] : [2]);
  visibilityFilter = ids;
  meshes.forEach((_, id) => visibility(id, ids.includes(id)));
  dirty = true;
};
$("focus").onclick = () => {
  const m = measurement(selected);
  let target,
    distance = 5;
  if (m.points_scene_mm) {
    const p = m.points_scene_mm.map(v);
    target = p[0].clone().add(p[1]).multiplyScalar(0.5);
    distance = Math.max(3, p[0].distanceTo(p[1]) * 2.5);
  } else {
    const mesh = meshes.get(m.label || 1);
    if (!mesh) return;
    const box = new THREE.Box3().setFromObject(mesh);
    target = box.getCenter(new THREE.Vector3());
    distance = box.getSize(new THREE.Vector3()).length() * 2;
  }
  const dir = camera.position.clone().sub(controls.target).normalize();
  controls.target.copy(target);
  camera.position.copy(target).addScaledVector(dir, distance);
  controls.update();
  dirty = true;
};
$("opacity").oninput = () =>
  meshes.forEach((m) => {
    m.material.opacity = Number($("opacity").value);
    m.material.transparent = m.material.opacity < 1;
    m.material.depthWrite = m.material.opacity === 1;
    m.material.needsUpdate = true;
  });
$("xray").onchange = () => select(selected);
document
  .querySelectorAll("[data-view]")
  .forEach((b) => (b.onclick = () => view(b.dataset.view)));
const axes = new THREE.AxesHelper(1);
axes.position.set(-2.6, -2.6, 3.6);
scene.add(axes);
new ResizeObserver(() => {
  const w = viewport.clientWidth,
    h = viewport.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  dirty = true;
}).observe(viewport);
controls.addEventListener("change", () => (dirty = true));
document.addEventListener("input", () => (dirty = true));
document.addEventListener("click", () => (dirty = true));
renderer.setAnimationLoop(() => {
  controls.update();
  if (!dirty) return;
  dirty = false;
  light.position.copy(camera.position);
  renderer.render(scene, camera);
  for (const { element, point } of labels) {
    const p = point.clone().project(camera);
    let occluded = false;
    if (!$("xray").checked && p.z >= -1 && p.z <= 1) {
      const direction = point.clone().sub(camera.position);
      const distance = direction.length();
      labelRay.set(camera.position, direction.normalize());
      labelRay.far = Math.max(0, distance - 0.05);
      occluded = labelRay
        .intersectObjects(
          [...meshes.values()].filter((m) => m.visible),
          false,
        )
        .some((hit) => !cutPlane || cutPlane.distanceToPoint(hit.point) >= 0);
    }
    element.style.display = p.z > 1 || p.z < -1 || occluded ? "none" : "block";
    element.style.left = ((p.x + 1) / 2) * viewport.clientWidth + 8 + "px";
    element.style.top = ((-p.y + 1) / 2) * viewport.clientHeight - 20 + "px";
  }
});
let sliceVersion = 0;
function shape() {
  return alignedSource() ? [300, 340, 420] : data.shape_zyx;
}
function configureSlice() {
  const a = Number($("axis").value);
  $("slice-index").max = shape()[a] - 1;
  $("slice-index").value = Math.min(
    Number($("slice-index").value),
    shape()[a] - 1,
  );
}
function slicePoint(coords) {
  if (alignedSource()) {
    const f = data.frames.tibia;
    coords = f.output_xyz_to_original_xyz_matrix.map((row, i) =>
      row.reduce(
        (sum, x, j) => sum + x * coords[j],
        f.output_xyz_to_original_xyz_offset[i],
      ),
    );
  }
  return world(coords);
}
async function updateSlice() {
  const version = ++sliceVersion,
    axis = Number($("axis").value),
    index = Number($("slice-index").value),
    lo = Number($("low").value),
    hi = Number($("high").value);
  $("slice-number").value = `${index} / ${$("slice-index").max}`;
  if (!(lo < hi)) {
    $("slice-status").textContent = "Upper HU must exceed lower HU.";
    return;
  }
  const q = new URLSearchParams({
    source: $("source").value,
    axis,
    index,
    low: lo,
    high: hi,
    overlay: $("overlay").checked ? "1" : "0",
  });
  $("slice-status").textContent = "Loading slice…";
  try {
    const res = await fetch("/api/slice?" + q);
    if (!res.ok) throw Error("Slice request failed");
    const blob = await res.blob();
    if (version !== sliceVersion) return;
    const url = URL.createObjectURL(blob),
      img = new Image();
    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = reject;
      img.src = url;
    });
    if (version !== sliceVersion) {
      URL.revokeObjectURL(url);
      return;
    }
    $("slice").src = url;
    $("slice").onload = () => URL.revokeObjectURL(url);
    $("slice-status").textContent = "";
    if (slicePlane) {
      scene.remove(slicePlane);
      slicePlane.geometry.dispose();
      slicePlane.material.map.dispose();
      slicePlane.material.dispose();
      slicePlane = null;
      dirty = true;
    }
    // Array planes retain ascending rows/columns, exactly as in the PNG.
    const sz = shape(),
      fixed = 2 - axis,
      others = [2, 1, 0].filter((i) => i !== fixed),
      row = others[0],
      col = others[1];
    const coords = [
      [0, 0],
      [1, 0],
      [1, 1],
      [0, 1],
    ].map(([c, r]) => {
      const xyz = [0, 0, 0];
      xyz[fixed] = index;
      xyz[col] = c * (sz[2 - col] - 1);
      xyz[row] = r * (sz[2 - row] - 1);
      return slicePoint(xyz);
    });
    cutPlane = null;
    if ($("cutaway").checked) {
      cutPlane = new THREE.Plane().setFromCoplanarPoints(
        ...coords.slice(0, 3).map(v),
      );
      if (cutPlane.distanceToPoint(camera.position) > 0) cutPlane.negate();
    }
    for (const mesh of meshes.values()) {
      mesh.material.clippingPlanes = cutPlane ? [cutPlane] : [];
      mesh.material.needsUpdate = true;
    }
    dirty = true;
    if ($("plane3d").checked) {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute(
        "position",
        new THREE.Float32BufferAttribute(coords.flat(), 3),
      );
      geo.setAttribute(
        "uv",
        new THREE.Float32BufferAttribute([0, 1, 1, 1, 1, 0, 0, 0], 2),
      );
      geo.setIndex([0, 1, 2, 0, 2, 3]);
      geo.computeVertexNormals();
      const texture = new THREE.Texture(img);
      texture.needsUpdate = true;
      texture.colorSpace = THREE.SRGBColorSpace;
      const mat = new THREE.MeshBasicMaterial({
        map: texture,
        side: THREE.DoubleSide,
      });
      slicePlane = new THREE.Mesh(geo, mat);
      scene.add(slicePlane);
      dirty = true;
    }
    $("slice-coordinate").textContent =
      `${alignedSource() ? "Aligned UVW" : "Original XYZ"} coordinates; fixed ${alignedSource() ? "WVU"[axis] : "ZYX"[axis]} = ${index}. 10.5 µm spacing.`;
  } catch (e) {
    if (version === sliceVersion) $("slice-status").textContent = e.message;
  }
}
for (const id of ["source", "axis"])
  $(id).onchange = () => {
    configureSlice();
    updateSlice();
  };
for (const id of ["overlay", "plane3d", "cutaway", "low", "high"])
  $(id).onchange = updateSlice;
let timer;
$("slice-index").oninput = () => {
  clearTimeout(timer);
  timer = setTimeout(updateSlice, 90);
};
for (const [id, delta] of [
  ["previous", -1],
  ["next", 1],
])
  $(id).onclick = () => {
    $("slice-index").value = Math.max(
      0,
      Math.min(
        Number($("slice-index").max),
        Number($("slice-index").value) + delta,
      ),
    );
    updateSlice();
  };
$("method").textContent =
  data.display +
  " Calibration: stored × 0.391063 − 1000 HU; median 3×3×3, mask >2500 HU, seeds 3000–5000 HU. Independent Astra review completed; anatomical acceptance remains provisional.";
const stages = [...new Set(data.evidence.map((e) => e.stage))];
for (const s of stages) {
  const option = document.createElement("option");
  option.value = s;
  option.textContent = s;
  $("stage").append(option);
}
$("stage").value = "review-10-measurements";
function evidence() {
  $("evidence").replaceChildren();
  for (const e of data.evidence.filter(
    (e) => e.stage === $("stage").value && e.name.endsWith(".png"),
  )) {
    const figure = document.createElement("figure"),
      a = document.createElement("a"),
      img = document.createElement("img"),
      caption = document.createElement("figcaption");
    a.href = "/assets/" + e.path;
    a.target = "_blank";
    img.src = a.href;
    img.loading = "lazy";
    img.alt = e.name;
    caption.textContent = e.name;
    a.append(img);
    figure.append(a, caption);
    $("evidence").append(figure);
  }
}
$("stage").onchange = evidence;
evidence();
for (const e of data.evidence.filter((e) => !e.name.endsWith(".png"))) {
  const a = document.createElement("a");
  a.href = "/assets/" + e.path;
  a.target = "_blank";
  a.textContent = e.stage + "/" + e.name;
  $("downloads").append(a);
}
mountParameterCatalog(catalog, data, select);
view("overview");
updateSlice();
const initialMeasurement = new URLSearchParams(location.search).get(
  "measurement",
);
const initialParameter =
  new URLSearchParams(location.search).get("parameter") || "4";
select(
  catalog.parameters.some((p) => String(p.id) === initialParameter)
    ? `deck-${initialParameter}`
    : "deck-4",
);

if (initialMeasurement && data.measurements[initialMeasurement])
  select(initialMeasurement);
try {
  for (const s of data.structures) {
    const res = await fetch("/assets/" + s.file);
    if (!res.ok) throw Error("Mesh unavailable: " + s.name);
    const buffer = await res.arrayBuffer();
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      "position",
      new THREE.BufferAttribute(new Float32Array(buffer, 0, s.vertices * 3), 3),
    );
    geometry.setIndex(
      new THREE.BufferAttribute(
        new Uint32Array(buffer, s.vertices * 12, s.indices),
        1,
      ),
    );
    geometry.computeVertexNormals();
    geometry.computeBoundingSphere();
    const material = new THREE.MeshStandardMaterial({
      color: s.color,
      clippingPlanes: cutPlane ? [cutPlane] : [],
      roughness: 0.85,
      metalness: 0,
      transparent: Number($("opacity").value) < 1,
      opacity: Number($("opacity").value),
      depthWrite: Number($("opacity").value) === 1,
      side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.userData.label = s.label;
    mesh.visible = !visibilityFilter || visibilityFilter.includes(s.label);
    scene.add(mesh);
    dirty = true;
    meshes.set(s.label, mesh);
    const row = document.createElement("div");
    row.className = "structure";
    const check = document.createElement("input");
    check.type = "checkbox";
    check.checked = mesh.visible;
    check.id = "structure-" + s.label;
    check.onchange = () => visibility(s.label, check.checked);
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = s.color;
    const label = document.createElement("label");
    label.htmlFor = check.id;
    label.textContent = s.name.replaceAll("-", " ");
    const small = document.createElement("small");
    small.textContent = s.volume_mm3.toFixed(3) + " mm³";
    const review = document.createElement("button");
    review.textContent = "Review";
    review.setAttribute("aria-label", "Review " + s.name);
    review.onclick = () => select("structure-" + s.label);
    row.append(check, swatch, label, small, review);
    $("structures").append(row);
    $("load").textContent =
      `Loaded ${meshes.size} / ${data.structures.length} structures`;
  }
  $("load").textContent = `${meshes.size} structures ready`;
  setTimeout(() => ($("load").style.display = "none"), 2000);
} catch (e) {
  $("load").textContent = "Could not load all structures: " + e.message;
  console.error(e);
}
// Expose read-only numerical state for browser verification, not mutation of scientific data.
window.reviewState = () => ({
  structures: meshes.size,
  visible: [...meshes].filter(([, m]) => m.visible).map(([id]) => id),
  selected,
  landmarks: labels.map((l) => l.point.toArray()),
  slicePlane: !!slicePlane,
  cutaway: !!cutPlane,
});
