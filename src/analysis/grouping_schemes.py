"""
Multi-level feature grouping for ADAE permutation importance.

Three complementary schemes (mutually exclusive within each scheme):

1. **domain** — four Sentinel-2 reflectance domains (water pixel, vegetation pixel,
   vegetation-corrected, water anomaly), each bundling raw bands + same-surface indices;
   plus spatial, temporal and meteorology.

2. **band** — twelve MSI band groups (B1–B12). Raw reflectances map to their band;
   multi-band indices are assigned to a **primary diagnostic band** (documented below)
   so each feature belongs to exactly one group for permutation.

3. **function** — process-oriented optics (chlorophyll/red-edge, turbidity/ISS–SPM,
   CDOM/organic, water/NDWI, vegetation context, broadband reflectance) plus
   spatial, temporal and meteorology.

Legacy 13-category taxonomy remains in feature_taxonomy.py for migration reference.
"""

from __future__ import annotations

from dataclasses import dataclass

# Primary MSI band for each index recipe (diagnostic / highest-weight band)
INDEX_RECIPE_PRIMARY_BAND: dict[str, str] = {
    "ndre": "B5",
    "ndwi": "B8",
    "b4_b5_ratio": "B5",
    "oci": "B3",
    "grvi": "B4",
    "nir_red": "B8",
    "spm_proxy": "B5",
    "flh": "B5",
    "cire": "B7",
    "cdom_proxy": "B1",
    "cdom_b1_b3": "B1",
    "turb_b4_b8": "B4",
    "multiband_iss": "B5",
    "b4_b8_prod": "B4",
    "red_edge_slope": "B7",
    "mndwi": "B11",
    "veg_ndvi": "B8",
}

_CHLORO_KEYS = ("ndre", "flh", "cire", "oci", "grvi", "aci")
_TURB_KEYS = (
    "spm", "turb", "mndwi", "multiband", "b4_b8", "red_edge",
    "b4_b5", "nir_red", "iss",
)
_CDOM_KEYS = ("cdom",)
_WATER_KEYS = ("ndwi",)


def _parse_index_recipe(name: str) -> str | None:
    if not name.startswith("idx_"):
        return None
    body = name[4:]
    if body.startswith("veg_"):
        return "veg_ndvi"
    for prefix in ("sta_", "vc_", "anom_"):
        if body.startswith(prefix):
            return body[len(prefix):]
    return body


def _parse_spec_band(name: str) -> str | None:
    if not name.startswith("spec_"):
        return None
    tail = name.split("_")[-1]
    if tail.startswith("B"):
        return tail
    return None


def _index_functional(name: str) -> str:
    low = name.lower()
    if "veg_ndvi" in low or name.startswith("spec_veg_"):
        return "func_vegetation"
    if any(k in low for k in _CDOM_KEYS):
        return "func_cdom"
    if any(k in low for k in _CHLORO_KEYS):
        return "func_chlorophyll"
    if any(k in low for k in _TURB_KEYS):
        return "func_turbidity"
    if any(k in low for k in _WATER_KEYS):
        return "func_water_ndwi"
    if name.startswith("spec_"):
        return "func_broadband_reflectance"
    return "func_other"


def _classify_domain(name: str) -> str:
    if name in ("lat", "lon"):
        return "spatial_position"
    if name.startswith("trend_") or name in ("month", "year", "season_sin", "season_cos"):
        return "temporal_trend"
    if name.startswith("meteo_") and "T2m" in name:
        return "meteo_temperature"
    if name.startswith("meteo_"):
        return "meteo_moisture_rad"
    if name.startswith("spec_station_") or name.startswith("idx_sta_"):
        return "domain_station"
    if name.startswith("spec_veg_corrected_") or name.startswith("idx_vc_"):
        return "domain_veg_corrected"
    if name.startswith("spec_veg_") or name == "idx_veg_ndvi":
        return "domain_veg_pixel"
    if name.startswith("spec_anomaly_") or name.startswith("idx_anom_"):
        return "domain_anomaly"
    return "func_other"


def _classify_band(name: str) -> str:
    if name in ("lat", "lon"):
        return "spatial_position"
    if name.startswith("trend_") or name in ("month", "year", "season_sin", "season_cos"):
        return "temporal_trend"
    if name.startswith("meteo_") and "T2m" in name:
        return "meteo_temperature"
    if name.startswith("meteo_"):
        return "meteo_moisture_rad"
    band = _parse_spec_band(name)
    if band:
        return f"band_{band}"
    recipe = _parse_index_recipe(name)
    if recipe and recipe in INDEX_RECIPE_PRIMARY_BAND:
        return f"band_{INDEX_RECIPE_PRIMARY_BAND[recipe]}"
    return "other_non_band"


def _classify_function(name: str) -> str:
    if name in ("lat", "lon"):
        return "spatial_position"
    if name.startswith("trend_") or name in ("month", "year", "season_sin", "season_cos"):
        return "temporal_trend"
    if name.startswith("meteo_") and "T2m" in name:
        return "meteo_temperature"
    if name.startswith("meteo_"):
        return "meteo_moisture_rad"
    return _index_functional(name)


SCHEME_META: dict[str, dict] = {
    "domain": {
        "title": "Reflectance domain",
        "description": "Four Sentinel-2 pixel domains + spatiotemporal / meteorology",
        "order": [
            "domain_station",
            "domain_veg_corrected",
            "domain_veg_pixel",
            "domain_anomaly",
            "spatial_position",
            "temporal_trend",
            "meteo_temperature",
            "meteo_moisture_rad",
            "func_other",
        ],
        "labels": {
            "domain_station": "Water-pixel domain",
            "domain_veg_corrected": "Veg-corrected domain",
            "domain_veg_pixel": "Vegetation-pixel domain",
            "domain_anomaly": "Water-anomaly domain",
            "spatial_position": "Spatial position",
            "temporal_trend": "Temporal / trend",
            "meteo_temperature": "Air temperature",
            "meteo_moisture_rad": "Soil moisture / precip. / radiation",
            "func_other": "Other",
        },
        "colors": {
            "domain_station": "#3182BD",
            "domain_veg_corrected": "#9ECAE1",
            "domain_veg_pixel": "#74C476",
            "domain_anomaly": "#C994C7",
            "spatial_position": "#1F4E79",
            "temporal_trend": "#4C78A8",
            "meteo_temperature": "#E9A663",
            "meteo_moisture_rad": "#F4C68A",
            "func_other": "#BDBDBD",
        },
        "classify": _classify_domain,
    },
    "band": {
        "title": "MSI band (primary)",
        "description": "Twelve Sentinel-2 bands; indices assigned to primary diagnostic band",
        "order": (
            [f"band_B{b}" for b in (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12)]
            + ["spatial_position", "temporal_trend", "meteo_temperature",
               "meteo_moisture_rad", "other_non_band"]
        ),
        "labels": {
            **{f"band_B{b}": f"Band B{b}" for b in (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12)},
            "spatial_position": "Spatial position",
            "temporal_trend": "Temporal / trend",
            "meteo_temperature": "Air temperature",
            "meteo_moisture_rad": "Soil moisture / precip. / radiation",
            "other_non_band": "Other (non-band)",
        },
        "colors": {
            **{f"band_B{b}": plt_c for b, plt_c in zip(
                (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12),
                ["#08306B", "#08519C", "#2171B5", "#4292C6", "#6BAED6",
                 "#9ECAE1", "#C6DBEF", "#DEEBF7", "#F7FBFF", "#FD8D3C",
                 "#E6550D", "#A63603"],
            )},
            "spatial_position": "#1F4E79",
            "temporal_trend": "#4C78A8",
            "meteo_temperature": "#E9A663",
            "meteo_moisture_rad": "#F4C68A",
            "other_non_band": "#BDBDBD",
        },
        "classify": _classify_band,
    },
    "function": {
        "title": "Process / function",
        "description": "Optical process groups + broadband reflectance + covariates",
        "order": [
            "func_chlorophyll",
            "func_turbidity",
            "func_cdom",
            "func_water_ndwi",
            "func_vegetation",
            "func_broadband_reflectance",
            "spatial_position",
            "temporal_trend",
            "meteo_temperature",
            "meteo_moisture_rad",
            "func_other",
        ],
        "labels": {
            "func_chlorophyll": "Chlorophyll / red-edge",
            "func_turbidity": "Turbidity / ISS–SPM",
            "func_cdom": "CDOM / organic optics",
            "func_water_ndwi": "Water / NDWI",
            "func_vegetation": "Vegetation context",
            "func_broadband_reflectance": "Broadband reflectance",
            "spatial_position": "Spatial position",
            "temporal_trend": "Temporal / trend",
            "meteo_temperature": "Air temperature",
            "meteo_moisture_rad": "Soil moisture / precip. / radiation",
            "func_other": "Other",
        },
        "colors": {
            "func_chlorophyll": "#74C476",
            "func_turbidity": "#CB4B16",
            "func_cdom": "#41AB5D",
            "func_water_ndwi": "#4292C6",
            "func_vegetation": "#A1D99B",
            "func_broadband_reflectance": "#9E9AC8",
            "spatial_position": "#1F4E79",
            "temporal_trend": "#4C78A8",
            "meteo_temperature": "#E9A663",
            "meteo_moisture_rad": "#F4C68A",
            "func_other": "#BDBDBD",
        },
        "classify": _classify_function,
    },
}

ATTR_SCHEMES = ("domain", "band", "function")
ATTR_PRIMARY_SCHEME = "function"


@dataclass(frozen=True)
class GroupingScheme:
    name: str
    title: str
    description: str
    order: tuple[str, ...]
    labels: dict[str, str]
    colors: dict[str, str]

    def classify(self, feature: str) -> str:
        fn = SCHEME_META[self.name]["classify"]
        return fn(feature)

    def category_members(self, fc: list[str]) -> dict[str, list[str]]:
        buckets: dict[str, list[str]] = {c: [] for c in self.order}
        for f in fc:
            cat = self.classify(f)
            if cat not in buckets:
                buckets[cat] = []
            buckets[cat].append(f)
        return {k: sorted(v) for k, v in buckets.items() if v}

    def category_indices(self, fc: list[str]) -> dict[str, list[int]]:
        members = self.category_members(fc)
        return {k: [fc.index(c) for c in v] for k, v in members.items()}


def get_scheme(name: str) -> GroupingScheme:
    if name not in SCHEME_META:
        raise KeyError(f"Unknown grouping scheme '{name}'; choose from {ATTR_SCHEMES}")
    meta = SCHEME_META[name]
    return GroupingScheme(
        name=name,
        title=meta["title"],
        description=meta["description"],
        order=tuple(meta["order"]),
        labels=meta["labels"],
        colors=meta["colors"],
    )


def scheme_csv_path(out_dir, scheme: str, target: str, protocol: str):
    from pathlib import Path
    return Path(out_dir) / scheme / f"category_{target}_{protocol}.csv"
