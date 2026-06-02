from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from microct_analysis.processing.dicom import LoadError, load_dicom


def _write_slice(
    path: Path,
    *,
    instance: int,
    z: float,
    value: int,
    modality: str = "CT",
    orientation: list[float] | None = None,
) -> None:
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = generate_uid()

    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.Modality = modality
    dataset.Manufacturer = "Synthetic"
    dataset.ManufacturerModelName = "UnitTest"
    dataset.AcquisitionDate = "20260504"
    dataset.InstanceNumber = instance
    dataset.ImagePositionPatient = [10.0, 20.0, z]
    dataset.ImageOrientationPatient = orientation or [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    dataset.PixelSpacing = [0.2, 0.3]
    dataset.SliceThickness = 0.5
    dataset.Rows = 2
    dataset.Columns = 3
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.RescaleSlope = 2
    dataset.RescaleIntercept = -10
    pixels = np.full((2, 3), value, dtype=np.uint16)
    dataset.PixelData = pixels.tobytes()
    dataset.save_as(path, write_like_original=False)


def test_load_dicom_sorts_by_slice_projection_and_converts_to_float32(tmp_path: Path) -> None:
    _write_slice(tmp_path / "slice_2.dcm", instance=2, z=0.5, value=10)
    _write_slice(tmp_path / "slice_1.dcm", instance=1, z=1.0, value=20)

    volume = load_dicom(tmp_path)

    assert volume.data.shape == (2, 2, 3)
    assert volume.data.dtype == np.float32
    assert np.all(volume.data[0] == 10.0)  # 10 * slope 2 + intercept -10
    assert np.all(volume.data[1] == 30.0)
    assert volume.spacing == (0.5, 0.2, 0.3)
    np.testing.assert_allclose(np.diag(volume.affine)[:3], [0.3, 0.2, 0.5])
    np.testing.assert_allclose(volume.affine[:3, 3], [10.0, 20.0, 0.5])
    assert volume.provenance["manufacturer"] == "Synthetic"
    assert volume.provenance["model"] == "UnitTest"
    assert volume.provenance["acquisition_date"] == "20260504"
    assert volume.provenance["slice_count"] == 2
    assert volume.provenance["n_slices"] == 2
    assert volume.provenance["original_spacing"] == [0.5, 0.2, 0.3]
    assert volume.provenance["slice_uid_hash"]
    assert volume.provenance["transfer_syntax_uid"]


def test_load_dicom_accepts_ima_and_suffixless_files(tmp_path: Path) -> None:
    _write_slice(tmp_path / "slice_1", instance=1, z=0.0, value=10)
    _write_slice(tmp_path / "slice_2.ima", instance=2, z=0.5, value=20)

    volume = load_dicom(tmp_path)

    assert volume.data.shape == (2, 2, 3)
    assert volume.spacing == (0.5, 0.2, 0.3)


def test_load_dicom_raises_on_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(LoadError) as excinfo:
        load_dicom(tmp_path)
    assert excinfo.value.flag == "missing-dicom-files"


def test_load_dicom_raises_on_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(LoadError) as excinfo:
        load_dicom(tmp_path / "missing")
    assert excinfo.value.flag == "missing-dicom-files"


def test_load_dicom_flags_non_uniform_spacing(tmp_path: Path) -> None:
    _write_slice(tmp_path / "slice_1.dcm", instance=1, z=0.0, value=10)
    _write_slice(tmp_path / "slice_2.dcm", instance=2, z=0.5, value=20)
    _write_slice(tmp_path / "slice_3.dcm", instance=3, z=2.0, value=30)

    with pytest.raises(LoadError) as excinfo:
        load_dicom(tmp_path)

    assert excinfo.value.flag == "non-uniform-spacing"


def test_load_dicom_flags_wrong_modality(tmp_path: Path) -> None:
    _write_slice(tmp_path / "slice_1.dcm", instance=1, z=0.0, value=10, modality="MR")

    with pytest.raises(LoadError) as excinfo:
        load_dicom(tmp_path)

    assert excinfo.value.flag == "wrong-modality"


def test_load_dicom_flags_bad_orientation(tmp_path: Path) -> None:
    path = tmp_path / "slice_1.dcm"
    _write_slice(path, instance=1, z=0.0, value=10)
    dataset = pydicom.dcmread(path)
    dataset.ImageOrientationPatient = []
    dataset.save_as(path, write_like_original=False)

    with pytest.raises(LoadError) as excinfo:
        load_dicom(tmp_path)

    assert excinfo.value.flag == "missing-dicom-tags"
