import re
from datetime import datetime

import h5py
import numpy as np


class ImarisReader:
    """
    A convenient and efficient reader for Imaris (.ims) HDF5 files.

    Attributes:
        filepath (str): Path to the .ims file.
        dtype (np.dtype): Data type of the image.
        shape (tuple): Dimensions (Time, Channels, Z, Y, X).
        voxel_size (tuple): (Z, Y, X) voxel size in microns.
        timestamps (list): List of datetime objects for each time point.
        channels_info (list): List of dicts containing metadata (Name, Wavelengths, Exposure).
        resolution_levels (int): Number of resolution levels available.
        n_channels (int): number of channels

    Methods:
        read(c=0, t=0, z=None, res_level=0)
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self._file = h5py.File(filepath, "r")

        # Initialize containers
        self.voxel_size = (1.0, 1.0, 1.0)
        self.timestamps = []
        self.channels_info = []
        self._level_shapes = {}  # res_level -> (Z, Y, X)

        # Run setup
        self._scan_structure()
        self._parse_metadata()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self._file:
            self._file.close()
            self._file = None

    def _decode_imaris_attribute(self, value):
        """
        Decodes the specific Imaris byte-array attribute format into a standard string.
        Examples:
            [b'6' b'0' b'0'] -> "600"
            "MyImage" -> "MyImage"
        """
        # Case 1: Already string/bytes scalar
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, str):
            return value

        # Case 2: Numpy array or list (The Imaris "Character Array")
        if isinstance(value, (np.ndarray, list, tuple)):
            val_array = np.array(value).flatten()
            if val_array.size == 0:
                return ""

            # Bytes/Strings elements
            if val_array.dtype.kind in ("S", "U") or isinstance(
                val_array[0], (bytes, np.bytes_)
            ):
                chars = []
                for x in val_array:
                    if isinstance(x, (bytes, np.bytes_)):
                        chars.append(x.decode("utf-8"))
                    else:
                        chars.append(str(x))
                return "".join(chars)

            # Integer elements (ASCII codes)
            if val_array.dtype.kind in ("i", "u"):
                return "".join([chr(x) for x in val_array])

            # Numeric elements (e.g. float array), return first as string
            return str(val_array[0])

        return str(value)

    def _get_val(self, grp, key, type_func=str):
        """
        Extracts attribute. Tries to cast to `type_func`.
        If casting fails (e.g. converting "600 nm" to float), returns the String.
        """
        if grp is None or key not in grp.attrs:
            return None

        raw_val = grp.attrs[key]
        str_val = self._decode_imaris_attribute(raw_val)

        if not str_val:
            return None

        # Try strict cast
        try:
            if type_func == bool:
                return str_val.lower() == "true"
            return type_func(str_val)
        except (ValueError, TypeError):
            # If cast fails (e.g. "600 nm"), return the original string
            return str_val

    def _scan_structure(self):
        """Dynamically scans the HDF5 hierarchy to determine dimensions."""
        dataset_root = self._file["DataSet"]

        # 1. Identify Resolution Levels
        self._res_groups = sorted(
            [
                k
                for k in dataset_root.keys()
                if "ResolutionLevel" in k and re.search(r"\d+", k)
            ],
            key=lambda x: int(re.search(r"\d+", x).group()),
        )
        self.resolution_levels = len(self._res_groups)

        if self.resolution_levels == 0:
            raise ValueError("No ResolutionLevels found in Imaris file.")

        # 2. Inspect Level 0 to find Time and Channels
        l0 = dataset_root[self._res_groups[0]]
        time_groups = [k for k in l0.keys() if "TimePoint" in k]

        if not time_groups:
            raise ValueError("No TimePoints found in Imaris file.")

        self.n_timepoints = len(time_groups)

        t0_name = sorted(
            time_groups,
            key=lambda x: int(re.search(r"\d+", x).group())
            if re.search(r"\d+", x)
            else 0,
        )[0]
        t0 = l0[t0_name]

        channel_groups = [k for k in t0.keys() if "Channel" in k]
        self.n_channels = len(channel_groups)

        # 3. Get Image Dimensions
        c0_name = sorted(
            channel_groups,
            key=lambda x: int(re.search(r"\d+", x).group())
            if re.search(r"\d+", x)
            else 0,
        )[0]
        c0 = t0[c0_name]
        data_node = c0["Data"]
        self.dtype = data_node.dtype

        # Dimensions are almost always pure integers, so strict int() works
        sx = self._get_val(data_node, "ImageSizeX", int)
        sy = self._get_val(data_node, "ImageSizeY", int)
        sz = self._get_val(data_node, "ImageSizeZ", int)

        # Fallback to dataset shape if attributes failed
        if not isinstance(sx, (int, float)):
            d_shape = data_node.shape
            if len(d_shape) == 3:
                sz, sy, sx = d_shape
            elif len(d_shape) == 2:
                sz = 1
                sy, sx = d_shape
            else:
                sx = d_shape[-1]
                sy = d_shape[-2]
                sz = d_shape[-3] if len(d_shape) > 2 else 1

        self.size_x = int(sx)
        self.size_y = int(sy)
        self.size_z = int(sz)
        self.shape = (
            self.n_timepoints,
            self.n_channels,
            self.size_z,
            self.size_y,
            self.size_x,
        )

    def _parse_metadata(self):
        """Parses global metadata from /DataSetInfo."""
        info_group = self._file.get("DataSetInfo")
        if not info_group:
            return

        # --- Voxel Size ---
        # Helper: ensures we have a float for math, defaulting to 0.0/1.0 if we get a string/None
        def ensure_float(v, default=0.0):
            if isinstance(v, (float, int)):
                return float(v)
            # If it's a string like "1.5", conversion works. If "1.5 um", it fails -> returns default.
            try:
                return float(v)
            except:
                return default

        img_info = info_group.get("Image")

        # Extents usually come as pure numbers ("0.0", "1024.5")
        min_x = ensure_float(self._get_val(img_info, "ExtMin0", float), 0.0)
        min_y = ensure_float(self._get_val(img_info, "ExtMin1", float), 0.0)
        min_z = ensure_float(self._get_val(img_info, "ExtMin2", float), 0.0)
        max_x = ensure_float(self._get_val(img_info, "ExtMax0", float), 1.0)
        max_y = ensure_float(self._get_val(img_info, "ExtMax1", float), 1.0)
        max_z = ensure_float(self._get_val(img_info, "ExtMax2", float), 1.0)

        vox_x = (max_x - min_x) / self.size_x if self.size_x > 0 else 1.0
        vox_y = (max_y - min_y) / self.size_y if self.size_y > 0 else 1.0
        vox_z = (max_z - min_z) / self.size_z if self.size_z > 0 else 1.0
        self.voxel_size = (vox_z, vox_y, vox_x)

        # --- Timestamps ---
        time_info = info_group.get("TimeInfo")
        if time_info:
            for i in range(self.n_timepoints):
                ts_str = None
                keys = [f"TimePoint{i + 1}", f"TimePoint {i + 1}"]
                for k in keys:
                    val = self._get_val(time_info, k, str)
                    if val:
                        ts_str = val
                        break

                if ts_str:
                    formats = ["%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"]
                    dt = None
                    for fmt in formats:
                        try:
                            dt = datetime.strptime(ts_str.strip(), fmt)
                            break
                        except ValueError:
                            continue
                    self.timestamps.append(dt)
                else:
                    self.timestamps.append(None)

        # --- Channel Info ---
        self.channels_info = []
        for i in range(self.n_channels):
            c_grp = info_group.get(f"Channel {i}")
            info = {
                "id": i,
                "name": f"Channel {i}",
                "emission_wavelength": None,
                "excitation_wavelength": None,
                "exposure_time": None,
            }

            if c_grp:
                info["name"] = (
                    self._get_val(c_grp, "Name", str) or f"Channel {i}"
                )

                # We try to get float, but if it is "600 nm", it returns "600 nm" (string)
                # We prefer LSMEmissionWavelength if available
                em = self._get_val(c_grp, "LSMEmissionWavelength", float)
                if em is None:
                    em = self._get_val(c_grp, "EmissionWavelength", float)
                info["emission_wavelength"] = em

                ex = self._get_val(c_grp, "LSMExcitationWavelength", float)
                if ex is None:
                    ex = self._get_val(c_grp, "ExcitationWavelength", float)
                info["excitation_wavelength"] = ex

                info["exposure_time"] = self._get_val(
                    c_grp, "ExposureTime", float
                )

            self.channels_info.append(info)

    def level_shape(self, res_level):
        """(Z, Y, X) image size at a resolution level."""
        if res_level not in self._level_shapes:
            data = self._dataset(0, 0, res_level)
            size = tuple(self._get_val(data, f"ImageSize{a}", int) for a in "ZYX")
            if not all(isinstance(v, int) for v in size):
                size = data.shape
            self._level_shapes[res_level] = size
        return self._level_shapes[res_level]

    def chunk_depth(self, res_level):
        """Number of Z planes per stored (compressed) chunk at a resolution level."""
        chunks = self._dataset(0, 0, res_level).chunks
        return chunks[0] if chunks else 1

    def read(self, c=0, t=0, z=None, res_level=0):
        """
        Reads image data.
        Args:
            c, t: Channel and Timepoint indices.
            z: Z-slice index or slice. None for full volume.
            res_level: Resolution level (0=Full).
        """
        if res_level >= self.resolution_levels:
            raise ValueError(f"Resolution level {res_level} unavailable.")

        dataset = self._dataset(c, t, res_level)

        # Stored datasets can be padded beyond the image size - crop to it
        sz, sy, sx = self.level_shape(res_level)
        if z is None:
            z = slice(0, sz)
        return dataset[z, :sy, :sx]

    def _dataset(self, c, t, res_level):
        """The HDF5 dataset holding channel c, timepoint t at a resolution level."""
        try:
            res_grp_name = self._res_groups[res_level]
            res_grp = self._file["DataSet"][res_grp_name]

            t_candidates = [
                k
                for k in res_grp.keys()
                if f"TimePoint {t}" in k or f"TimePoint{t}" == k
            ]
            if not t_candidates:
                raise ValueError(f"TimePoint {t} not found")
            t_grp = res_grp[t_candidates[0]]

            c_candidates = [
                k
                for k in t_grp.keys()
                if f"Channel {c}" in k or f"Channel{c}" == k
            ]
            if not c_candidates:
                raise ValueError(f"Channel {c} not found")

            dataset = t_grp[c_candidates[0]]["Data"]
        except Exception as e:
            raise ValueError(f"Error locating data: {str(e)}")

        return dataset

    def __repr__(self):
        return (
            f"<ImarisReader: {self.filepath}\n"
            f"  Shape (T,C,Z,Y,X): {self.shape}\n"
            f"  Dtype: {self.dtype}\n"
            f"  Voxel Size (um): {self.voxel_size}\n"
            f"  Channels: {[c['name'] for c in self.channels_info]}>"
        )


if __name__ == "__main__":
    print("Imaris Reader Module Loaded.")
