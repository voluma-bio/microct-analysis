// Presentation definitions are kept separate from computed sample measurements.
export function resolveParameter(parameter, data) {
  const prior = parameter.source_measurement
    ? data.measurements[parameter.source_measurement]
    : null;
  const candidate =
    parameter.status === "Candidate label"
      ? data.structures.find((s) => s.label === parameter.labels[0])
      : null;
  return {
    ...(prior || {}),
    parameter,
    title: `${parameter.id}. ${parameter.name}`,
    definition: parameter.definition,
    label: parameter.labels[0],
    labels: parameter.labels,
    prior_value: prior?.value ?? candidate?.volume_mm3 ?? null,
    prior_unit: prior?.unit ?? (candidate ? "mm^3" : null),
  };
}

export function formatRelated(parameter, data) {
  const p = resolveParameter(parameter, data);
  if (p.prior_value === null) return "Not measured";
  const unit =
    p.prior_unit === "mm^3"
      ? " mm³"
      : p.prior_unit === "1"
        ? ""
        : ` ${p.prior_unit}`;
  return `${p.prior_value.toFixed(3)}${unit} (${parameter.status === "Candidate label" ? "candidate label" : "prior estimate"})`;
}

export function mountParameterCatalog(catalog, data, select) {
  const derived = catalog.derived[0];
  document.getElementById("derived-parameter").textContent =
    `${derived.name}: ${derived.status}. ${derived.definition} ${derived.method}`;
  const picker = document.getElementById("parameter-picker");
  let group;
  for (const parameter of catalog.parameters) {
    if (!group || group.label !== parameter.group) {
      group = document.createElement("optgroup");
      group.label = parameter.group;
      picker.append(group);
    }
    const option = document.createElement("option");
    option.value = `deck-${parameter.id}`;
    option.textContent = `${parameter.id}. ${parameter.name}`;
    group.append(option);
    const row = document.createElement("tr");
    const name = document.createElement("td");
    const button = document.createElement("button");
    button.textContent = `${parameter.id}. ${parameter.name}`;
    button.onclick = () => {
      select(`deck-${parameter.id}`);
      document
        .getElementById("details")
        .scrollIntoView({ behavior: "smooth", block: "center" });
    };
    name.append(button);
    const status = document.createElement("td");
    status.textContent = parameter.status;
    const value = document.createElement("td");
    value.textContent = formatRelated(parameter, data);
    const slide = document.createElement("td");
    const link = document.createElement("a");
    link.href = `/assets/evidence/parameter-deck/slide-${parameter.slide}.png`;
    link.target = "_blank";
    link.textContent = `Slide ${parameter.slide}`;
    slide.append(link);
    row.append(name, status, value, slide);
    document.getElementById("parameter-rows").append(row);
  }
  picker.onchange = () => select(picker.value);
  document.getElementById("vertebral-note").textContent =
    catalog.vertebral.definition;
  const vertebral = document.getElementById("vertebral-rows");
  for (const id of catalog.vertebral.parameter_ids) {
    const p = catalog.parameters.find((p) => p.id === id);
    const li = document.createElement("li");
    li.textContent = `${id}. ${p.name.replace("Trabecular ROI ", "").replace("Trabecular ROI", "ROI")}: unavailable; no vertebral dataset.`;
    vertebral.append(li);
  }
}

export function appendParameterDetails(m, data, showAnatomy) {
  if (!m.parameter) return;
  const p = m.parameter,
    details = document.getElementById("details");
  const status = document.createElement("p");
  status.className = "parameter-status";
  status.textContent = `${p.status} · ${p.unit}`;
  details.append(status);
  const related = document.createElement("p");
  related.textContent = `Related result: ${formatRelated(p, data)}. This is not a newly accepted measurement under the slide definition.`;
  details.append(related);
  const method = document.createElement("p");
  method.textContent = p.method;
  details.append(method);
  const button = document.createElement("button");
  button.textContent = p.context
    ? "Inspect tibial context"
    : "Isolate linked structures";
  button.onclick = showAnatomy;
  details.append(button);
  const figure = document.createElement("a");
  figure.href = `/assets/evidence/parameter-deck/slide-${p.slide}.png`;
  figure.target = "_blank";
  const img = document.createElement("img");
  img.className = "parameter-slide";
  img.src = figure.href;
  img.alt = `Source slide ${p.slide}, open to inspect full-size annotations`;
  figure.append(img);
  details.append(figure);
  if (p.context) {
    const note = document.createElement("p");
    note.textContent =
      "The tibia shown is anatomical context only. The requested plate, growth-plate surface or trabecular ROI has not been segmented.";
    details.append(note);
  }
}
