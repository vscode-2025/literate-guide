"""Streamlit UI for exploring crimes near TTC stops.

Relies on the analytics functions defined in `crime_around_stops.py`.
The app supports two workflows:
- Use the repo's default GTFS + crime files (paths prefilled below).
- Upload custom CSVs for stops, stop_times (optional), and crime data.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from crime_around_stops import (
    aggregate,
    attach_nearest_stop,
    load_crime,
    load_stop_times,
    load_stops,
    make_map,
)


# ----------------------------
# Helpers
# ----------------------------
def _load_csv_from_uploader(file) -> pd.DataFrame:
    """Read uploaded CSV/TSV with pandas, handling gzip if needed."""
    if file is None:
        return pd.DataFrame()
    # streamlit provides a BytesIO-like object; sniff delimiter quickly
    buffer = io.BytesIO(file.getvalue())
    # Try comma first, fall back to tab
    try:
        return pd.read_csv(buffer)
    except Exception:
        buffer.seek(0)
        return pd.read_csv(buffer, sep="\t")


def _load_events_json(path: str | Path) -> tuple[set[pd.Timestamp], pd.DataFrame]:
    """
    Flatten the City events feed into:
    - a set of event dates (date only) for labeling crimes as event-day
    - a dataframe of point events with lat/lon (when present) for optional map overlay
    """
    if not path or not Path(path).exists():
        return set(), pd.DataFrame()

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    records = raw["value"] if isinstance(raw, dict) and "value" in raw else raw
    events = pd.DataFrame(records)

    if "event_dates" in events.columns:
        date_series = pd.json_normalize(events["event_dates"].explode())["date"]
        event_days = set(pd.to_datetime(date_series, unit="ms", errors="coerce").dropna().dt.normalize().dt.date)
    else:
        event_days = set()

    # Points: take first gps_lat/gps_lng per event when available
    if "event_locations" in events.columns:
        exploded = events["event_locations"].explode().reset_index().rename(columns={"index": "event_idx"})
        locs = pd.json_normalize(exploded["event_locations"])

        def _parse_gps(val):
            if not isinstance(val, str) or not val.strip():
                return None
            try:
                obj = json.loads(val)
                if isinstance(obj, list) and obj and isinstance(obj[0], dict):
                    return obj
            except Exception:
                return None
            return None

        locs["gps"] = locs["location_gps"].apply(_parse_gps)
        locs["lat"] = locs["gps"].apply(lambda x: x[0].get("gps_lat") if isinstance(x, list) and x else None)
        locs["lon"] = locs["gps"].apply(lambda x: x[0].get("gps_lng") if isinstance(x, list) and x else None)
        locs["event_idx"] = exploded["event_idx"]
        locs = locs.dropna(subset=["lat", "lon"])
        if not locs.empty:
            locs["event_name"] = locs["event_idx"].map(events["event_name"])
            locs = locs[["event_name", "lat", "lon"]]
        else:
            locs = pd.DataFrame()
    else:
        locs = pd.DataFrame()
    return event_days, locs


@st.cache_data(show_spinner=False)
def load_data(
    stops_path: Optional[str],
    stop_times_path: Optional[str],
    crime_path: Optional[str],
    events_path: Optional[str],
    radius_m: float,
    divisions: list[str],
    years_back: int,
    freq: str,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, set[pd.Timestamp], pd.DataFrame]:
    """Load, filter, link, and aggregate data for the visualization."""
    if stops_path is None or crime_path is None:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), set(), pd.DataFrame()

    stops = load_stops(stops_path)
    crime = load_crime(crime_path)

    # Date filtering: exact range if provided, otherwise years_back
    if start_date is not None or end_date is not None:
        if start_date is not None:
            crime = crime[crime["crime_dt"] >= pd.Timestamp(start_date)]
        if end_date is not None:
            # include the whole end date
            crime = crime[crime["crime_dt"] < pd.Timestamp(end_date) + pd.Timedelta(days=1)]
    else:
        cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(years=years_back)
        crime = crime[crime["crime_dt"] >= cutoff]
    crime = crime[crime["division"].isin(divisions)]

    stop_times_summary = None
    if stop_times_path and Path(stop_times_path).exists():
        stop_times_summary = load_stop_times(stop_times_path)

    linked = attach_nearest_stop(crime, stops, max_dist_m=radius_m)
    stops_for_map = stops[stops["stop_id"].isin(linked["nearest_stop_id"])].copy()

    if stop_times_summary is not None and not stop_times_summary.empty:
        linked = linked.merge(
            stop_times_summary,
            left_on="nearest_stop_id",
            right_on="stop_id",
            how="left",
            suffixes=("", "_stoptime"),
        )

    agg = aggregate(linked, freq=freq)
    event_days, event_points = _load_events_json(events_path) if events_path else (set(), pd.DataFrame())
    if event_days:
        linked["date"] = linked["crime_dt"].dt.date
        linked["label"] = np.where(linked["date"].isin(event_days), "Event day", "Normal day")
    return linked, agg, stops_for_map, event_days, event_points


def compute_risk_scores(
    linked: pd.DataFrame,
    stops_for_map: pd.DataFrame,
    intensity_w: float,
    severity_w: float,
    divisions_filter: Optional[list[str]] = None,
    crime_types_filter: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Compute baseline risk components and overall score per stop.
    Assumes `linked` already has crimes within 250m and date filtering applied.
    """
    if linked.empty or stops_for_map.empty:
        return pd.DataFrame()

    df = linked.copy()

    # Apply filters
    if divisions_filter:
        df = df[df["division"].isin(divisions_filter)]
    type_col = None
    for cand in ["MCI", "crime_type", "MCI_CATEGORY"]:
        if cand in df.columns:
            type_col = cand
            break
    if crime_types_filter and type_col:
        df = df[df[type_col].isin(crime_types_filter)]
    if df.empty:
        return pd.DataFrame()

    # Time window for temporal risk
    df["hour"] = df["crime_dt"].dt.hour

    # Intensity: crimes per month
    months_span = df["crime_dt"].dt.to_period("M")
    min_per, max_per = months_span.min(), months_span.max()
    months_range = (max_per.year * 12 + max_per.month) - (min_per.year * 12 + min_per.month) + 1
    months_range = max(int(months_range), 1)
    stop_counts = df.groupby("nearest_stop_id").size().rename("crime_count")
    intensity = stop_counts / months_range

    # Severity
    severity_weights = {
        "Assault": 3,
        "Robbery": 3,
        "Break and Enter": 2,
        "Break & Enter": 2,
        "Auto Theft": 2,
        "Theft Over": 1,
    }
    if type_col:
        df["severity_w"] = df[type_col].map(severity_weights).fillna(1.0)
    else:
        df["severity_w"] = 1.0
    severity = (df.groupby("nearest_stop_id")["severity_w"].mean()).rename("severity_raw")

    # Combine components
    risk_df = pd.concat([intensity, severity], axis=1).fillna(0).reset_index()
    risk_df = risk_df.rename(columns={"nearest_stop_id": "stop_id"})

    def minmax(series):
        if series.empty:
            return series
        min_v, max_v = series.min(), series.max()
        if max_v == min_v:
            return pd.Series(0.0, index=series.index)
        return (series - min_v) / (max_v - min_v)

    risk_df["intensity_component"] = minmax(risk_df["crime_count"])
    risk_df["severity_component"] = minmax(risk_df["severity_raw"])
    weight_sum = intensity_w + severity_w
    if weight_sum <= 0:
        intensity_w, severity_w = 0.7, 0.3
        weight_sum = 1.0
    intensity_w, severity_w, temporal_w = (
        intensity_w / weight_sum,
        severity_w / weight_sum,
        0.0,
    )

    risk_df["baseline_risk_score"] = (
        0.5 * 0  # placeholder to keep structure explicit
        + intensity_w * risk_df["intensity_component"]
        + severity_w * risk_df["severity_component"]
    )

    # Attach names and coordinates
    risk_df = risk_df.merge(
        stops_for_map[["stop_id", "stop_name", "stop_lat", "stop_lon"]],
        on="stop_id",
        how="left",
    )

    cols = [
        "stop_id",
        "stop_name",
        "stop_lat",
        "stop_lon",
        "baseline_risk_score",
        "intensity_component",
        "severity_component",
        "crime_count",
    ]
    return risk_df[cols].sort_values("baseline_risk_score", ascending=False)


def render_map(
    linked: pd.DataFrame,
    stops_for_map: pd.DataFrame,
    freq: str,
    radius_m: float,
    event_points: pd.DataFrame | None = None,
    risk_lookup: dict[str, float] | None = None,
    risk_level_lookup: dict[str, str] | None = None,
    daily_avg_lookup: dict[str, float] | None = None,
    event_daily_avg_lookup: dict[str, float] | None = None,
    normal_daily_avg_lookup: dict[str, float] | None = None,
    majority_type_lookup: dict[str, str] | None = None,
):
    if linked.empty or stops_for_map.empty:
        st.info("Load data to view the map.")
        return
    fmap = make_map(
        linked,
        stops_for_map,
        freq=freq,
        radius_m=radius_m,
        risk_lookup=risk_lookup,
        risk_level_lookup=risk_level_lookup,
        daily_avg_lookup=daily_avg_lookup,
        event_daily_avg_lookup=event_daily_avg_lookup,
        normal_daily_avg_lookup=normal_daily_avg_lookup,
        majority_type_lookup=majority_type_lookup,
    )
    if event_points is not None and not event_points.empty:
        import folium

        layer = folium.FeatureGroup(name="Events", show=True)
        for _, row in event_points.iterrows():
            folium.CircleMarker(
                location=[row.lat, row.lon],
                radius=6,
                color="#7b1fa2",
                fill=True,
                fill_opacity=0.8,
                popup=row.event_name,
            ).add_to(layer)
        layer.add_to(fmap)
        folium.LayerControl(collapsed=False).add_to(fmap)
    html = fmap.get_root().render()
    st.components.v1.html(html, height=700, scrolling=True)


# ----------------------------
# UI
# ----------------------------
def main():
    st.set_page_config(page_title="Crimes near TTC stops", layout="wide")
    st.title("Crimes near TTC stops")
    st.write(
        "Link TPS Major Crime Indicators to TTC stops and explore patterns by period, "
        "division, and distance."
    )

    default_stops = "Complete GTFS/stops.txt"
    default_stop_times = "data/processed/stop_times_with_stops.csv.gz"
    default_crime = "Major_Crime_Indicators.csv"
    default_events = "Festivals and events json feed.json"

    sidebar = st.sidebar
    with sidebar:
        st.header("Data sources")
        use_defaults = st.checkbox(
            "Use repo defaults",
            value=Path(default_stops).exists() and Path(default_crime).exists(),
            help="Use the included GTFS stops and TPS crime export.",
            key="sidebar_use_repo_defaults",
        )

        stops_path = default_stops if use_defaults else None
        crime_path = default_crime if use_defaults else None
        stop_times_path = default_stop_times if use_defaults else None
        events_path = default_events if use_defaults and Path(default_events).exists() else None

        if not use_defaults:
            stops_file = st.file_uploader("stops.txt (CSV/TSV)", type=["csv", "txt", "tsv"], key="upload_stops")
            crime_file = st.file_uploader("Crime CSV/TSV", type=["csv", "txt", "tsv"], key="upload_crime")
            stop_times_file = st.file_uploader("stop_times (optional)", type=["csv", "txt", "tsv"], key="upload_stop_times")
            events_file = st.file_uploader("Events JSON", type=["json"], key="upload_events")

            # Save uploaded files to temp buffers for loader compatibility
            if stops_file:
                df = _load_csv_from_uploader(stops_file)
                tmp = Path(stops_file.name)
                df.to_csv(tmp, index=False)
                stops_path = str(tmp)
            if crime_file:
                df = _load_csv_from_uploader(crime_file)
                tmp = Path(crime_file.name)
                df.to_csv(tmp, index=False)
                crime_path = str(tmp)
            if stop_times_file:
                df = _load_csv_from_uploader(stop_times_file)
                tmp = Path(stop_times_file.name)
                df.to_csv(tmp, index=False)
                stop_times_path = str(tmp)
            if events_file:
                tmp = Path(events_file.name)
                tmp.write_bytes(events_file.getvalue())
                events_path = str(tmp)

        st.header("Filters")
        radius_m = st.slider("Max distance to link crimes to a stop (m)", 50, 500, 250, step=25, key="filter_radius_m")
        freq = st.radio("Aggregation", options=["M", "Y"], format_func=lambda x: "Monthly" if x == "M" else "Yearly", key="filter_freq")
        years_back = st.slider("Years back from today", 1, 10, 1, key="filter_years_back")
        use_date_range = st.checkbox("Filter by exact date range", value=True, key="filter_use_date_range")
        start_date = end_date = None
        if use_date_range:
            today = pd.Timestamp.today().normalize().date()
            default_start = (pd.Timestamp(today) - pd.DateOffset(years=years_back)).date()
            date_sel = st.date_input(
                "Date range (inclusive)",
                value=(default_start, today),
                key="filter_date_range",
            )
            if isinstance(date_sel, tuple) and len(date_sel) == 2:
                start_date, end_date = date_sel
            elif isinstance(date_sel, (pd.Timestamp, pd.DatetimeIndex)):
                start_date = end_date = date_sel
            else:
                start_date = end_date = None
        st.subheader("Risk weights")
        intensity_w = st.slider("Intensity weight", 0.0, 1.0, 0.7, 0.05, key="risk_intensity_w")
        severity_w = st.slider("Severity weight", 0.0, 1.0, 0.3, 0.05, key="risk_severity_w")
        st.subheader("Division filter")
        division_select = st.multiselect("Divisions", options=["14", "51", "52"], default=["14", "51", "52"], key="filter_divisions")
        # placeholder for crime type selector (filled after data load)
        crime_type_placeholder = st.empty()

    # Load once per parameter combo
    linked, agg, stops_for_map, event_days, event_points = load_data(
        stops_path=stops_path,
        stop_times_path=stop_times_path,
        crime_path=crime_path,
        events_path=events_path,
        radius_m=radius_m,
        divisions=division_select or ["14", "51", "52"],
        years_back=years_back,
        start_date=start_date,
        end_date=end_date,
        freq=freq,
    )

    # Crime type selector now that we know the column/values
    crime_type_col = None
    for cand in ["MCI", "crime_type", "MCI_CATEGORY"]:
        if cand in linked.columns:
            crime_type_col = cand
            break
    if crime_type_col:
        type_options = sorted(linked[crime_type_col].dropna().unique())
        selected_types = crime_type_placeholder.multiselect("Crime types", options=type_options, default=type_options, key="filter_crime_types")
    else:
        selected_types = None

    if linked.empty:
        st.warning("No data loaded. Check file paths or upload required CSVs.")
        return

    # Compute risk once so it can be shown on map and in tables
    risk_df = compute_risk_scores(
        linked,
        stops_for_map,
        intensity_w=intensity_w,
        severity_w=severity_w,
        divisions_filter=division_select,
        crime_types_filter=selected_types,
    )

    # Risk levels: even 0.25-width bins across score range
    if not risk_df.empty:
        bins = [-float("inf"), 0.25, 0.50, 0.75, float("inf")]
        labels = ["Low", "Moderate", "Elevated", "High"]
        risk_df["risk_level"] = pd.cut(risk_df["baseline_risk_score"], bins=bins, labels=labels, include_lowest=True)
    else:
        risk_df["risk_level"] = pd.Series(dtype="object")

    risk_lookup = None
    risk_level_lookup = None
    daily_avg_lookup = None
    majority_type_lookup = None
    event_daily_avg_lookup = None
    normal_daily_avg_lookup = None
    if not risk_df.empty:
        risk_lookup = dict(zip(risk_df["stop_id"].astype(str), risk_df["baseline_risk_score"]))
        risk_level_lookup = dict(zip(risk_df["stop_id"].astype(str), risk_df["risk_level"]))
    if not linked.empty:
        days_range = (linked["crime_dt"].max().normalize() - linked["crime_dt"].min().normalize()).days + 1
        days_range = max(days_range, 1)
        daily_avg = linked.groupby("nearest_stop_id").size() / days_range
        daily_avg_lookup = dict(zip(daily_avg.index.astype(str), daily_avg.values))
        # majority crime type per stop
        type_col_for_majority = None
        for cand in ["MCI", "crime_type", "MCI_CATEGORY"]:
            if cand in linked.columns:
                type_col_for_majority = cand
                break
        if type_col_for_majority:
            maj = (
                linked.groupby(["nearest_stop_id", type_col_for_majority])
                .size()
                .reset_index(name="n")
                .sort_values(["nearest_stop_id", "n"], ascending=[True, False])
                .drop_duplicates("nearest_stop_id")
                .set_index("nearest_stop_id")[type_col_for_majority]
            )
            majority_type_lookup = dict(zip(maj.index.astype(str), maj.values))
        if "label" in linked.columns:
            daily = linked.copy()
            daily["date"] = daily["crime_dt"].dt.date
            daily = (
                daily.groupby(["nearest_stop_id", "label", "date"])
                .size()
                .rename("count")
                .reset_index()
            )
            label_avg = (
                daily.groupby(["nearest_stop_id", "label"])["count"]
                .mean()
                .reset_index()
                .pivot(index="nearest_stop_id", columns="label", values="count")
            )
            if "Event day" in label_avg.columns:
                event_daily_avg_lookup = dict(zip(label_avg.index.astype(str), label_avg["Event day"].fillna(0).values))
            if "Normal day" in label_avg.columns:
                normal_daily_avg_lookup = dict(zip(label_avg.index.astype(str), label_avg["Normal day"].fillna(0).values))

    cols = st.columns([2, 1])
    with cols[0]:
        st.subheader("Map")
        render_map(
            linked,
            stops_for_map,
            freq=freq,
            radius_m=radius_m,
            event_points=event_points,
            risk_lookup=risk_lookup,
            risk_level_lookup=risk_level_lookup,
            daily_avg_lookup=daily_avg_lookup,
            event_daily_avg_lookup=event_daily_avg_lookup,
            normal_daily_avg_lookup=normal_daily_avg_lookup,
            majority_type_lookup=majority_type_lookup,
        )
    with cols[1]:
        st.subheader("Summary")
        st.metric("Linked crimes", len(linked))
        st.metric("Stops with crimes", linked["nearest_stop_id"].nunique() if not linked.empty else 0)
        if not agg.empty:
            st.write("Crime counts by period and stop")
            st.dataframe(agg.head(1000))
            csv_bytes = agg.to_csv(index=False).encode("utf-8")
            st.download_button("Download aggregated CSV", data=csv_bytes, file_name="crime_stop_agg.csv", key="dl_agg_csv")
        if event_days:
            st.write(f"Event days loaded: {len(event_days)}")
            st.write(f"Events with coordinates: {len(event_points)}")

    if risk_df.empty:
        st.info("No data available for risk scoring with current filters.")
    else:
        top_n = max(10, int(np.ceil(0.1 * len(risk_df))))
        top_df = risk_df.nlargest(top_n, "baseline_risk_score")

        st.write(f"Top {top_n} high-risk stops")
        table_cols = [
            "stop_name",
            "baseline_risk_score",
            "intensity_component",
            "severity_component",
            "risk_level",
            "crime_count",
        ]
        table = top_df[table_cols].copy()
        table[["baseline_risk_score", "intensity_component", "severity_component"]] = table[
            ["baseline_risk_score", "intensity_component", "severity_component"]
        ].round(3)
        table["crime_count"] = table["crime_count"].astype(int)
        st.dataframe(table)

        # Bar chart of highest-risk stops
        chart = (
            alt.Chart(top_df.head(20))
            .mark_bar()
            .encode(
                x=alt.X("baseline_risk_score:Q", title="Baseline risk score"),
                y=alt.Y("stop_name:N", sort="-x", title="Stop"),
                color=alt.Color("risk_level:N", title="Risk level"),
                tooltip=[
                    "stop_name",
                    alt.Tooltip("baseline_risk_score:Q", format=".3f"),
                    alt.Tooltip("intensity_component:Q", format=".3f"),
                    alt.Tooltip("severity_component:Q", format=".3f"),
                    "risk_level:N",
                    "crime_count:Q",
                ],
            )
            .properties(height=400)
        )
        st.altair_chart(chart, use_container_width=True)

        # Component breakdown for selected stop
        selected_stop = st.selectbox(
            "Inspect components for stop", options=risk_df["stop_name"], index=0 if not risk_df.empty else None, key="risk_selected_stop"
        )
        breakdown = risk_df[risk_df["stop_name"] == selected_stop].iloc[0]
        comp_df = pd.DataFrame(
            {
                "component": ["Intensity", "Severity"],
                "value": [
                    breakdown["intensity_component"],
                    breakdown["severity_component"],
                ],
            }
        )
        comp_chart = (
            alt.Chart(comp_df)
            .mark_bar()
            .encode(
                x=alt.X("component:N", title="Component"),
                y=alt.Y("value:Q", title="Normalized value", scale=alt.Scale(domain=[0, 1])),
                color="component",
            )
            .properties(height=250)
        )
        st.altair_chart(comp_chart, use_container_width=True)

        # Download risk table
        st.download_button(
            "Download risk scores CSV",
            data=risk_df.round(4).to_csv(index=False).encode("utf-8"),
            file_name="baseline_stop_risk.csv",
            key="dl_risk_csv",
        )

    # ----------------------------
    # Event vs Normal Day Comparison
    # ----------------------------
    st.divider()
    if "label" in linked.columns:
        st.subheader("Event vs Normal Day Comparison")
        daily = linked.copy()
        daily["date"] = daily["crime_dt"].dt.date
        daily = (
            daily.groupby(["nearest_stop_id", "nearest_stop_name", "label", "date"])
            .size()
            .rename("count")
            .reset_index()
        )
        avg = (
            daily.pivot_table(
                index=["nearest_stop_id", "nearest_stop_name"],
                columns="label",
                values="count",
                aggfunc="mean",
                fill_value=0,
            )
            .reset_index()
        )
        # Ensure columns exist
        if "Event day" not in avg.columns:
            avg["Event day"] = 0
        if "Normal day" not in avg.columns:
            avg["Normal day"] = 0
        avg = avg.rename(columns={"Event day": "event_avg", "Normal day": "normal_avg"})
        avg["delta_event_minus_normal"] = avg["event_avg"] - avg["normal_avg"]
        avg["total_avg"] = avg["event_avg"] + avg["normal_avg"]
        avg["event_share"] = np.where(avg["total_avg"] > 0, avg["event_avg"] / avg["total_avg"], 0.0)
        avg_display = avg.sort_values("delta_event_minus_normal", ascending=False)

        # KPIs
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Avg crimes/day on event days (mean across stops)", f"{avg['event_avg'].mean():.2f}")
        k2.metric("Avg crimes/day on normal days (mean across stops)", f"{avg['normal_avg'].mean():.2f}")
        k3.metric("Avg delta (event - normal)", f"{avg['delta_event_minus_normal'].mean():.2f}")
        k4.metric("Avg event share", f"{avg['event_share'].mean():.2f}")

        # Display sorted by delta descending
        avg_display = avg.sort_values("delta_event_minus_normal", ascending=False)
        top_n_ev = 20

        st.write("Average crimes per day by stop:")
        st.dataframe(
            avg_display.head(50)[
                ["nearest_stop_name", "event_avg", "normal_avg", "event_share", "delta_event_minus_normal"]
            ]
            .rename(columns={"nearest_stop_name": "stop_name"})
            .style.format({"event_avg": "{:.2f}", "normal_avg": "{:.2f}", "event_share": "{:.2f}", "delta_event_minus_normal": "{:.2f}"})
        )

        # Delta chart
        chart_df = avg_display.head(top_n_ev).rename(columns={"nearest_stop_name": "stop_name"})
        if not chart_df.empty:
            max_abs = chart_df["delta_event_minus_normal"].abs().max()
            max_abs = max(max_abs, 0.01)
            delta_chart = (
                alt.Chart(chart_df)
                .mark_bar()
                .encode(
                    x=alt.X("delta_event_minus_normal:Q", title="Event - Normal (avg crimes/day)"),
                    y=alt.Y("stop_name:N", sort="-x", title="Stop"),
                    color=alt.Color(
                        "delta_event_minus_normal:Q",
                        scale=alt.Scale(scheme="redblue", domain=[-max_abs, max_abs]),
                    ),
                    tooltip=[
                        "stop_name",
                        alt.Tooltip("event_avg:Q", format=".3f", title="Event avg"),
                        alt.Tooltip("normal_avg:Q", format=".3f", title="Normal avg"),
                        alt.Tooltip("event_share:Q", format=".2f", title="Event share"),
                        alt.Tooltip("delta_event_minus_normal:Q", format=".3f", title="Delta"),
                    ],
                )
                .properties(height=400)
            )
            st.altair_chart(delta_chart, use_container_width=True)

        st.download_button(
            "Download event vs normal table",
            data=avg_display.rename(columns={"nearest_stop_name": "stop_name"}).round(4).to_csv(index=False).encode("utf-8"),
            file_name="event_vs_normal.csv",
            key="dl_event_vs_normal_csv",
        )
    else:
        st.subheader("Event vs Normal Day Comparison")
        st.info(
            "Event vs Normal Day comparison is not available because no events data was loaded. "
            "To enable this feature, ensure the events JSON file (e.g. Festivals and events json feed.json) "
            "is available and contains valid event dates."
        )


if __name__ == "__main__":
    main()
