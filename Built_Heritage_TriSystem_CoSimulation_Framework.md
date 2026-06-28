# A Tri-System Co-Simulation Framework for Heritage Sites

## Visitor Flow, Urban Microclimate, and Deterioration-Maintenance Planning — Johor Bahru Heritage Research Programme

This framework extends the existing Johor Bahru heritage research pipeline — Space Syntax morphology mapping, Copernicus/ERA5 microclimate data, and the validated XGBoost/SHAP maintenance-priority model — from a single climate-only predictive chain into a three-system co-simulation. The goal is the next step explicitly named as future work in the existing research programme: a living, continuously updated decision-support layer (a "heritage digital twin") in which visitor behaviour, environmental stress, and material deterioration are modelled as interacting systems rather than separate studies. It is grounded throughout in the six Johor Bahru heritage sites already used as the research programme's case base: Bangunan Stesen Keretapi, Bangunan Sultan Ibrahim, Galeri Tenun, Johor Ancient Temple, Masjid India, and Masjid Sultan Abu Bakar.

---

## 1. Tool Landscape: Comparative Assessment

| Tool / Platform | Primary subsystem role | Core method | Typical outputs | Strengths for heritage research | Limitations | License / cost | Recommended role here |
|---|---|---|---|---|---|---|---|
| **SUMO** | Visitor flow (vehicular feeder) | Microscopic road-traffic simulation; pedestrians supported as a simplified secondary class | Vehicle/person counts, travel times, emissions | Open source; strong Python (TraCI) API for co-simulation; good for tour-bus/parking access modelling | Pedestrian behaviour model is rudimentary compared to dedicated pedestrian tools; not designed for crowd dynamics inside heritage precincts | Free, open source | Secondary — models vehicular/tour-bus access on the surrounding road network and converts arrivals into pedestrian-generation rates at heritage entrances |
| **AnyLogic** | Visitor flow (integration backbone) | Multi-method simulation: agent-based + discrete-event + system dynamics | Agent trajectories, density, queue lengths, custom KPIs | GIS-map import, pedestrian library, Java/Python connectivity, can host custom logic that calls external models | Commercial license for full features; steeper learning curve than single-purpose tools | Commercial (free Personal Learning Edition; paid Professional/University editions) | Co-simulation orchestrator — combines pedestrian agents with dwell-time/queueing logic and exchanges data with the microclimate and deterioration models |
| **PTV Viswalk** | Visitor flow (pedestrian microsimulation) | Social Force Model (Helbing–Molnár), validated pedestrian dynamics | Density, flow, dwell time, level-of-service per zone | Industry-standard, well-validated against real pedestrian counts; integrates with PTV Vissim for combined vehicle+pedestrian studies | Commercial, costly; less flexible for custom rule injection than AnyLogic | Commercial (academic licenses available) | Primary fine-grained pedestrian engine for heritage interiors, courtyards, and entrance chokepoints |
| **ENVI-met** | Urban microclimate | 3D prognostic CFD model of surface–vegetation–atmosphere interaction | Air temperature, RH, wind field, mean radiant/surface temperature | Research-grade, widely used in urban-heritage microclimate studies; resolves building-geometry effects on local climate | Computationally expensive at fine resolution; long run times limit scenario count | Commercial with academic pricing; limited free version | Detailed CFD engine for the urban-canyon microclimate around each heritage cluster |
| **Ladybug Tools** (Ladybug + Honeybee + Dragonfly) | Urban microclimate (BIM-coupled) | Grasshopper/Rhino plug-in suite wrapping EnergyPlus, Radiance, and the Urban Weather Generator | Solar radiation, shading, outdoor thermal comfort (UTCI), wind rose, per-facade exposure | Open source; couples directly to BIM/HBIM geometry; fast enough for many design-stage scenarios; good EPW handling from ERA5/NASA POWER | Requires Rhino/Grasshopper (commercial CAD host); less rigorous CFD than ENVI-met for complex airflow | Free (Ladybug Tools) / commercial host (Rhino) | Facade-resolved environmental-exposure engine operating directly on the HBIM model; complements ENVI-met for rapid scenario screening |
| **HBIM** (Revit/ArchiCAD + heritage libraries) | Cross-cutting spatial-semantic backbone | Parametric 3D modelling enriched with historic-building component libraries and condition metadata | Component geometry, material, construction era, condition history | Bridges geometry, material science, and sensor/ML data in one digital-twin-ready structure; supports point-cloud/laser-scan input | Labour-intensive to build for under-documented assets; interoperability with simulation tools needs IFC export discipline | Commercial authoring tools; methodology itself is license-agnostic | Holds component-level metadata, receives model predictions as visual attributes, and supplies geometry to the microclimate models |
| **GIS** (QGIS/ArcGIS) + Space Syntax engine (depthmapX) | Cross-cutting integration layer | Vector/raster spatial analysis; segment/axial angular-choice analysis (UCL depthmapX) | Integration, NACH, connectivity, zonal statistics, interpolated climate/visitor grids | Already the working toolchain in this programme (OSM → QGIS cleaning → depthmapX segment analysis); free and reproducible | depthmapX itself does not simulate dynamic movement — only static configurational potential | Free/open source (QGIS, depthmapX); ArcGIS commercial alternative | The shared georeferenced backbone aligning Space Syntax, visitor density, microclimate grids, and maintenance dashboards |
| **Python ML stack** (scikit-learn, XGBoost, SHAP, LIME) | Deterioration & maintenance prediction | Gradient-boosted ensemble learning with post-hoc explainability | Maintenance-priority class/score, feature-importance, extracted rules | Already validated on this exact case base (XGBoost accuracy 0.895, weighted F1 ≈ 0.87–0.88); transparent, reproducible in Jupyter | Performance bounded by training-data size and label quality; explainability tools approximate, not exact, causal mechanisms | Free, open source | Terminal predictive and explainability engine; consumes outputs of both other subsystems as engineered features |

---

## 2. Subsystem Models: Inputs, Outputs, Calibration, Validation, Linkage

### 2.1 People Movement / Visitor-Flow Subsystem

| Dimension | Specification |
|---|---|
| **Suitable simulator(s)** | PTV Viswalk as the primary social-force pedestrian microsimulation for movement inside and around heritage footprints; AnyLogic as the multi-method orchestration layer combining pedestrian agents with dwell-time and queueing logic and exchanging data with the other two subsystems; SUMO as a secondary feeder model converting vehicular/tour-bus arrivals on the surrounding road network into pedestrian-generation rates at heritage entrances. All three draw on a static configurational layer — Space Syntax segment analysis in depthmapX (Integration, NACH, Connectivity) — computed from the same OSM-to-depthmapX workflow already used for the Jalan Wong Ah Fook / Jalan Tun Abdul Razak / Jalan Siu Chin heritage core, which supplies the spatial-attraction weights that bias agent route choice. |
| **Required input data** | Street and building network geometry (OSM, cleaned in QGIS); per-segment Integration/NACH values; heritage site footprints, entrances and exits; observed visitor counts and trajectories (manual gate counts, CCTV, or the YOLOv11n+ECA edge-detection model already trained on 17,401 person-detection images for Jetson Nano deployment); dwell-time samples; opening hours, festival and tour-bus schedules; visitor-experience mining from Google Reviews/TripAdvisor for qualitative route and dwell preferences. |
| **Output variables** | Time-stepped visitor density per zone/component (persons/m²); dwell time per location (seconds–minutes); flow rate at chokepoints (persons/min); queue length at entrances; cumulative footfall-exposure load per heritage component over a defined interval (person-hours/m²/week) — the key variable passed downstream. |
| **Calibration method** | Tune social-force parameters (desired walking speed, personal-space radius, route-choice attraction weights derived from Integration/NACH) so simulated counts and densities at calibration checkpoints match YOLOv11n+ECA-detected counts, using the GEH statistic or RMSE as the acceptance criterion (commonly GEH < 5, standard transport-microsimulation practice). The configurational layer itself is calibrated first by checking the Spearman correlation between Integration/NACH and observed footfall, replicating the multi-scale validation already demonstrated for Hangzhou metro stations (strongest explanatory power at the R2000 urban scale). |
| **Validation method** | Hold-out validation against an independent time window or heritage site not used in calibration (e.g., calibrate on five of the six Johor Bahru sites and validate on the sixth, or calibrate on weekday patterns and validate on weekend/festival peaks); compare simulated versus observed density heat-maps and dwell-time distributions against on-site observation logs. |
| **Linkage to next subsystem** | Cumulative footfall-exposure load and dwell-time-weighted density per component become the engineered "Visitor Pressure Index," joining the existing microclimate feature set as inputs to the deterioration/maintenance-priority model; the same outputs are spatially overlaid (via the shared GIS/HBIM backbone) on the microclimate grid to flag zones facing combined footfall and environmental stress. |

### 2.2 Urban Microclimate Subsystem

| Dimension | Specification |
|---|---|
| **Suitable simulator(s)** | ENVI-met for fine-resolution 3D urban-canyon CFD simulation around each heritage cluster (captures building-geometry-modulated air temperature, relative humidity, wind field, and surface/mean radiant temperature); Ladybug Tools (Ladybug + Honeybee + Dragonfly) operating directly on HBIM-derived geometry for per-facade solar radiation, shading, wind-driven-rain potential, and outdoor thermal comfort — better suited to rapid scenario screening and direct coupling with the HBIM component database; GIS for spatially interpolating point-based reanalysis/station data into a continuous microclimate grid across the heritage precinct. |
| **Required input data** | Building and terrain geometry (HBIM/GIS); envelope material and albedo properties; vegetation and ground-cover data; boundary meteorological conditions from the Copernicus Climate Data Store ERA5 reanalysis (1940–2023, the same record already used in the existing pipeline), supplemented by NASA POWER for solar radiation where ERA5 resolution is coarse; local station observations (MetMalaysia Senai) for boundary-condition refinement and bias correction. |
| **Output variables** | Hourly/sub-hourly air temperature (°C), relative humidity (%), wind speed (m/s) and direction, precipitation (mm), surface/mean radiant temperature per facade, solar radiation and shading per surface, and derived indices such as a simplified Mould Growth Index and a wind-driven-rain exposure index per facade. |
| **Calibration method** | Adjust boundary conditions, envelope material parameters, and roughness/turbulence settings until simulated hourly temperature and relative humidity match a representative on-site or station monitoring period (typically 1–2 weeks per season), assessed via RMSE, mean bias error, and Willmott's index of agreement. |
| **Validation method** | Independent validation against a separate monitoring period or station not used in calibration; cross-check simulated long-run statistics against the historical ERA5 record already validated for this case base (≈2,400 mm annual rainfall, relative humidity routinely above 85%, mean temperature 26–27 °C for Bangunan Stesen Keretapi). |
| **Linkage to next subsystem** | Facade- and component-resolved environmental variables replace or augment the single-point historical climate series currently used in the XGBoost maintenance-priority model, enabling component-level rather than building-level prediction; the same variables are spatially joined with the visitor-pressure layer to build the combined exposure surface used for risk-threshold triggering. |

### 2.3 Heritage Deterioration and Maintenance-Planning Subsystem

| Dimension | Specification |
|---|---|
| **Suitable simulator(s)** | The Python ML stack (scikit-learn, XGBoost, SHAP, LIME) as the predictive and explainability engine — the same model family already validated on this case base; HBIM as the semantic, component-level data structure storing material, construction, and condition-history metadata, which also receives model predictions back as visualised attributes; GIS as the site-to-city integration layer aggregating component-level predictions across all six sites and linking them to the Power BI dashboard. |
| **Required input data** | Historical and simulated microclimate variables (Section 2.2); visitor-pressure variables (Section 2.1); heritage condition reports and inspection records (KALAM); building component metadata from HBIM (material, construction era, prior interventions); the existing labelled dataset of maintenance-priority ratings used to train and benchmark the baseline model. |
| **Output variables** | Predicted maintenance-priority class or risk score per component (0–1, or ordinal low/medium/high/urgent); SHAP/LIME feature-contribution values; extracted plain-language decision rules; ranked maintenance-priority list across the six sites. |
| **Calibration method** | Grid-search hyperparameter tuning (e.g., XGBoost learning rate ∈ {0.01, 0.05, 0.1}, n_estimators ∈ {50, 100, 200}, as in the existing pipeline); k-fold cross-validation during training; feature engineering informed by conservation science, including lag variables, seasonal indicators, the Mould Growth Index, and the new Visitor Pressure Index. |
| **Validation method** | Held-out test-set evaluation on accuracy, weighted F1, precision and recall, benchmarked against the existing climate-only result (XGBoost accuracy 0.895, weighted F1 ≈ 0.87–0.88, ahead of Random Forest 0.871/0.85, Decision Tree 0.825/0.81, RNN 0.834/0.82, and SVM 0.798/0.78); temporal hold-out (train on earlier years, test on the most recent years) to mimic real forecasting conditions; expert validation of SHAP/LIME-derived rules against KALAM conservation-practitioner judgment, as already practised for this case base. |
| **Linkage to next subsystem** | This is the terminal analytic node. Its risk scores feed a rule-based decision layer (Data → Analytics → Decision → Visualisation, the existing four-layer Decision-Support Framework) that triggers maintenance actions when predicted risk crosses defined thresholds; its outputs are written back into the HBIM condition database and the Power BI dashboard, closing the loop. |

---

## 3. Integrated Co-Simulation Framework

### 3.1 Coupling logic

Three engineered indices connect the subsystems without collapsing them into a single opaque score:

- **Visitor Pressure Index (VPI)** — a function of visitor density × dwell time, integrated over a defined time window per heritage component, normalised 0–1 across the six sites. Produced by Subsystem 1.
- **Environmental Stress Index (ESI)** — a composite of humidity-exceedance, temperature-exceedance, rainfall frequency, and wind speed, informed by (but not fixed to) the SHAP-derived contribution weights already discovered in the existing model (humidity ≈32%, temperature ≈27%, rainfall ≈20%, wind ≈14%). Produced by Subsystem 2.
- **Composite Maintenance Risk Score (MRS)** — learned, not hand-imposed: VPI and ESI enter the XGBoost model in Subsystem 3 as engineered features alongside the existing condition and lag variables, and the model itself learns how they combine to predict maintenance priority.

Maintenance actions are triggered when the MRS — or a specific extracted rule, such as sustained humidity above 80% combined with high rainfall ("moisture-risk flag"), or sustained high wind speed combined with elevated temperature ("facade-risk flag") — crosses a threshold calibrated against historical urgent-maintenance events. This operationalises, at a finer spatial and temporal grain, the configuration → movement → exposure → maintenance-load → decision chain already established conceptually in this research programme.

### 3.2 Orchestration and data exchange

- **Common spatial backbone**: a GIS database (PostGIS/GeoPackage) holding georeferenced layers for the Space Syntax segment network, heritage site footprints (from HBIM/IFC export), the microclimate grid (ENVI-met/Ladybug output rasters), and visitor-density rasters (PTV Viswalk/AnyLogic output) — functioning as the heritage digital-twin backbone this programme has already identified as its long-term direction.
- **Temporal alignment**: pedestrian microsimulation runs at second-level resolution but is aggregated to hourly/daily visitor-pressure summaries; microclimate simulation runs at hourly resolution; the deterioration model consumes monthly/seasonal aggregates, matching the lag and seasonal feature engineering already used in the baseline pipeline. A Python orchestration layer (pandas/xarray) performs this temporal resampling and assembles the joint feature table.
- **Software interfaces**: SUMO via its TraCI Python API; AnyLogic via Java/Python connectivity and database or REST exchange; PTV Viswalk via COM/.NET interfaces or exported simulation logs; ENVI-met via NetCDF/raster output read with xarray/rasterio; Ladybug Tools, itself Python-native, callable outside Grasshopper through its core libraries; HBIM exposed via IFC export consumed by GIS/Python (ifcopenshell). All feature assembly and the XGBoost/SHAP modelling run in a single Python/Jupyter environment, consistent with the existing workflow.

### 3.3 Feedback loop

Maintenance actions and updated condition assessments feed back into the HBIM condition database and the model's training dataset, enabling periodic retraining. This converts the framework from a one-off predictive study into a continuously learning decision-support system — the practical, data-engineering step toward the heritage digital twin already named as the closing direction of this research programme.

---

## 4. Open Data Sources

| Source | Data type | Coverage | Relevant subsystem(s) | Access / notes |
|---|---|---|---|---|
| **OpenStreetMap (OSM)** | Street and building network, land use | Global, continuously updated | Visitor flow (Space Syntax/depthmapX input), GIS backbone | Free; accessed via `osmnx`, the same toolchain already used to extract the Johor Bahru heritage core |
| **Copernicus Climate Data Store / ERA5 reanalysis** | Temperature, relative humidity, wind, precipitation | Global, hourly, 1940–present | Microclimate (boundary conditions); Deterioration (historical training features) | Free with registration; the climate source already underlying the existing XGBoost study |
| **NASA POWER** | Solar radiation, surface meteorology | Global, daily, 1981–present | Microclimate (EPW file generation for Ladybug/ENVI-met); cross-check for ERA5 | Free, API-accessible |
| **Malaysia heritage registers** (Jabatan Warisan Negara, Yayasan Warisan Johor, MBJB listings) | Heritage designation, building inventories | National/state | Deterioration (asset inventory); HBIM (baseline metadata) | Request-based; note the six Johor Bahru sites are nationally/state-listed, not UNESCO-inscribed — Melaka and George Town hold that status. UNESCO/ICCROM climate-risk guidance for World Heritage is still useful methodologically even though it is not a direct local data source for these sites |
| **MetMalaysia local weather stations** (e.g., Senai) | Ground-station meteorological observations | Local, hourly | Microclimate calibration and validation | Request-based; used to bias-correct ERA5 and validate ENVI-met/Ladybug outputs |
| **GIS basemap/cadastral data** (JUPEM, MBJB planning GIS) | Parcels, road centrelines, zoning | Local | GIS backbone, HBIM siting | Request-based, state/agency-held |
| **KALAM heritage condition reports and field surveys** | Component-level condition and defect ratings | Site-specific, periodic | Deterioration model training/validation; HBIM condition metadata | Institutional; already the labelled dataset behind the existing studies |
| **Roboflow "People Detection" dataset + on-site edge detections** | Annotated person-detection imagery; real-time visitor counts | Site-specific | Visitor-flow calibration and validation | Open dataset (17,401 images: 15,210 train / 1,431 validation / 760 test), plus the on-site Jetson Nano deployment |
| **Visitor-review mining** (Google Reviews, TripAdvisor) | Qualitative visitor experience and sentiment | Site-specific, continuous | Visitor-flow validation (route/dwell preference); tourism planning | Public; scraped as part of the existing GIS/AR heritage-trail project |

---

## 5. Deliverables

### 5.1 Simulation Architecture Diagram (text form)

```
                         OPEN & FIELD DATA SOURCES
   OSM | ERA5/Copernicus | NASA POWER | MetMalaysia stations
   KALAM condition reports | Heritage registers | YOLO/Roboflow imagery
   Visitor-review mining (Google Reviews / TripAdvisor)
                                   |
                                   v
        +------------------------------------------------------+
        |   SHARED SPATIAL BACKBONE: GIS + HBIM                |
        |   (digital-twin layer: PostGIS/GeoPackage + IFC)      |
        +------------------------------------------------------+
              |                     |                     |
              v                     v                     v
  +-----------------------+ +----------------------+ +------------------------+
  | SUBSYSTEM 1            | | SUBSYSTEM 2          | | SUBSYSTEM 3            |
  | VISITOR FLOW           | | URBAN MICROCLIMATE   | | DETERIORATION &        |
  |                        | |                      | | MAINTENANCE            |
  | Space Syntax (depthmapX| | ENVI-met (CFD) +     | | Python ML stack        |
  | Integration / NACH)    | | Ladybug/Honeybee/    | | (XGBoost + SHAP/LIME)  |
  |        |               | | Dragonfly            | |          ^             |
  |        v               | |        |             | |          |             |
  | PTV Viswalk / AnyLogic  | |        v             | |          |             |
  | pedestrian agents       | | Facade-resolved      | |          |             |
  | (+ SUMO vehicular feeder| | temp / RH / wind /   | |          |             |
  | for site access)        | | precip / solar / MGI | |          |             |
  +-----------+-------------+ +----------+-----------+ +------------------------+
              |                          |                         ^
              | Visitor Pressure         | Environmental           |
              | Index (density x         | Stress Index            |
              | dwell time)               | (humidity/temp/         |
              |                           | rain/wind weighted)     |
              +------------+--------------+-------------------------+
                           |
                           v
        +------------------------------------------------------+
        | COUPLING LAYER: feature assembly (Python/pandas/      |
        | xarray) -- temporal resampling + spatial join on GIS  |
        +------------------------------------------------------+
                           |
                           v
        +------------------------------------------------------+
        | DECISION LAYER: Composite Maintenance Risk Score +    |
        | rule-based threshold triggers                          |
        | (Data -> Analytics -> Decision, existing 4-layer DSF) |
        +------------------------------------------------------+
                           |
                           v
        +------------------------------------------------------+
        | VISUALISATION LAYER: Power BI / GIS dashboard for      |
        | KALAM, MBJB, Yayasan Warisan Johor, site managers      |
        +------------------------------------------------------+
                           |
                           v
              MAINTENANCE ACTION / CONSERVATION SCHEDULING
                           |
                           v
        (feedback loop) --> updates HBIM condition database
                              and retrains Subsystem 3 periodically
```

### 5.2 Table of Variables

| Variable | Symbol | Unit | Subsystem | Type | Source / model |
|---|---|---|---|---|---|
| Street/building network geometry | — | — | Visitor flow | Input | OSM, cleaned in QGIS |
| Space Syntax Integration | Int [HH] | dimensionless | Visitor flow | Input (derived) | depthmapX |
| Normalised Angular Choice | NACH | dimensionless | Visitor flow | Input (derived) | depthmapX |
| Visitor detection count | N | persons/interval | Visitor flow | Calibration target / output | YOLOv11n+ECA edge detection |
| Visitor density | ρ_v | persons/m² | Visitor flow | Output | PTV Viswalk / AnyLogic |
| Dwell time | T_d | seconds–minutes | Visitor flow | Output | PTV Viswalk / AnyLogic |
| Flow rate at chokepoint | Q | persons/min | Visitor flow | Output | PTV Viswalk / AnyLogic / SUMO |
| Visitor Pressure Index | VPI | dimensionless (0–1) | Coupling | Derived | ρ_v × T_d, normalised |
| Air temperature | T_a | °C | Microclimate | Input/Output | ERA5 / ENVI-met / Ladybug |
| Relative humidity | RH | % | Microclimate | Input/Output | ERA5 / ENVI-met |
| Wind speed | v_w | m/s | Microclimate | Input/Output | ERA5 / ENVI-met |
| Precipitation | P | mm | Microclimate | Input/Output | ERA5 / Copernicus |
| Surface / mean radiant temperature | T_s | °C | Microclimate | Output | ENVI-met / Ladybug |
| Solar radiation / shading | Rad | W/m² | Microclimate | Output | Ladybug / Honeybee |
| Mould Growth Index | MGI | index | Microclimate | Derived feature | Engineered (existing pipeline) |
| Environmental Stress Index | ESI | dimensionless (0–1) | Coupling | Derived | Weighted humidity/temp/rain/wind |
| Building component metadata | — | categorical | Deterioration | Input | HBIM |
| Historical condition rating | — | ordinal | Deterioration | Input | KALAM condition reports |
| Maintenance-priority prediction | MPR | class / probability (0–1) | Deterioration | Output | XGBoost |
| SHAP feature contribution | — | % of model variance | Deterioration | Output | SHAP |
| Model accuracy | Acc | % | Deterioration | Validation metric | Held-out test set |
| Weighted F1-score | F1 | — | Deterioration | Validation metric | Held-out test set |
| Composite Maintenance Risk Score | MRS | dimensionless (0–1) | Coupling/Output | Output | XGBoost, trained on VPI + ESI + condition features |
| Maintenance trigger threshold | θ | dimensionless | Decision layer | Calibrated parameter | ROC analysis / expert rule alignment |

### 5.3 Step-by-Step Methodology (Academic Paper Methods Section)

1. **Study area and case selection.** Define the six Johor Bahru heritage sites as the case population; select one site with the richest existing data (Bangunan Stesen Keretapi) for a detailed pilot coupling before scaling the framework to all six.
2. **Data acquisition and preprocessing.** Extract street and building network geometry from OSM via `osmnx`; clean in QGIS; retrieve ERA5 (Copernicus) and NASA POWER records; digitise KALAM condition reports; construct or import HBIM models for each site.
3. **Space Syntax configurational analysis.** Run segment analysis in depthmapX (Integration, NACH, Connectivity) at multiple radii (600 m–10,000 m), following the multi-scale approach already validated for metro-station accessibility; test the relationship between configurational measures and any available historical footfall proxy.
4. **Visitor-flow microsimulation development.** Build the pedestrian agent model in PTV Viswalk/AnyLogic using the cleaned network geometry, with route choice weighted by Integration/NACH; add SUMO to generate vehicular/tour-bus arrivals at site entrances.
5. **Visitor-flow calibration and validation.** Calibrate social-force parameters against YOLOv11n+ECA-detected counts at chokepoints using GEH/RMSE acceptance criteria; validate on a held-out site or time period.
6. **Microclimate model development.** Build the ENVI-met domain and/or Ladybug/Honeybee model from HBIM geometry; set boundary conditions from ERA5/NASA POWER; run for representative seasonal periods.
7. **Microclimate calibration and validation.** Calibrate against MetMalaysia station data using RMSE, mean bias error, and Willmott's index of agreement; validate against an independent period or station.
8. **Feature engineering and coupling.** Compute the Visitor Pressure Index and Environmental Stress Index; merge with the existing condition/microclimate dataset; align all variables to a common spatial (component-level) and temporal (monthly/seasonal) resolution via the GIS/HBIM backbone.
9. **Deterioration/maintenance-priority model training.** Train XGBoost and comparison models (Decision Tree, Random Forest, SVM, RNN) on the expanded feature set; tune hyperparameters by grid search with k-fold cross-validation.
10. **Model evaluation.** Benchmark against the existing climate-only result (accuracy 0.895, weighted F1 ≈ 0.87–0.88) to test whether the added visitor-pressure features improve performance; report accuracy, weighted F1, precision, recall.
11. **Explainability and rule extraction.** Apply SHAP/LIME; extract human-readable rules; validate them against KALAM expert judgment.
12. **Risk-threshold definition and decision-layer design.** Define maintenance-trigger thresholds via ROC analysis and alignment with extracted rules; implement the Data → Analytics → Decision → Visualisation dashboard in Power BI.
13. **Scenario and sensitivity analysis.** Vary visitor-flow calibration parameters and microclimate boundary conditions within plausible ranges; assess the robustness of site-priority rankings (e.g., Kendall's tau across scenarios).
14. **Stakeholder and expert validation.** Present integrated outputs to KALAM, MBJB, and site managers for face validity and decision-usability feedback.
15. **Reporting and dissemination.** Document the coupled pipeline, datasets, and code for reproducibility; report findings against the research questions and hypotheses below.

### 5.4 Possible Research Questions and Hypotheses

**RQ1.** To what extent does spatially resolved visitor pressure — derived from Space Syntax configuration and real-time edge-AI detection — improve the predictive performance of machine-learning maintenance-priority models beyond microclimate variables alone?
**H1.** Heritage components in zones with high Integration/NACH and high detected visitor density exhibit significantly higher predicted maintenance-priority scores than components in low-traffic zones, after controlling for microclimate exposure.

**RQ2.** Does coupling fine-scale, building-resolved microclimate simulation (ENVI-met/Ladybug) improve prediction accuracy over the single-point ERA5 series used in the existing studies?
**H2.** Facade-resolved, CFD-simulated microclimate variables explain significantly more variance in component-level deterioration risk than single-point reanalysis variables.

**RQ3.** Does adding a visitor-pressure feature preserve or improve the interpretability and trust of SHAP/LIME-based rule extraction for heritage managers, relative to the existing climate-only model?
**H3.** Heritage-conservation experts rate the interpretability and decision-usability of the expanded model's extracted rules at least as highly as those of the climate-only baseline.

**RQ4.** Do risk-threshold-triggered maintenance schedules generated by the coupled framework reduce the lag between deterioration onset and intervention compared with current periodic/reactive inspection cycles?
**H4.** A threshold-triggered, co-simulation-informed maintenance schedule produces a significantly shorter simulated time-to-intervention than a reactive baseline calibrated on historical inspection intervals.

**RQ5.** How sensitive is the predicted maintenance-priority ranking across the six Johor Bahru heritage sites to calibration uncertainty in the visitor-flow and microclimate subsystems?
**H5.** The ranking of top-priority sites remains stable (high Kendall's tau) across plausible parameter ranges in visitor-flow and microclimate calibration, indicating the integrated model is not unduly sensitive to reasonable calibration uncertainty.

### 5.5 Limitations and Validation Strategy

**Limitations.** Ground-truth data remain scarce at the component level: the existing studies rely on ERA5 reanalysis and periodic KALAM condition reports rather than dense on-site sensor networks, and on-site instrumentation is constrained by conservation rules that prohibit invasive fixings on century-old fabric. The six heritage sites differ materially and typologically (colonial-era railway station, royal building, temple, mosque, textile gallery), so a single trained model may not transfer cleanly across them — the existing studies already flag the exclusion of building-ontology features (material composition, construction typology, maintenance history) as a limitation. Coupling three simulators with different native time steps (seconds for pedestrian agents, hours for microclimate, months/seasons for deterioration) introduces aggregation choices that can shape results and must be reported transparently. Training data for "urgent" maintenance cases are likely to be small and imbalanced, as in the existing climate-only model. Expert validation through KALAM, while valuable, risks circularity if the same institution both supplies the labels and validates the rules; independent second-opinion review should be sought where possible. Finally, the framework is calibrated for Johor Bahru's equatorial climate and urban morphology and should not be assumed to generalise to other climatic or typological contexts without re-calibration.

**Validation strategy.** Validation should proceed in tiers, mirroring how each subsystem is already validated individually before integration:

| Tier | What is validated | Method |
|---|---|---|
| Component | Each subsystem's outputs in isolation | Visitor flow: GEH/RMSE against YOLO-detected counts. Microclimate: RMSE/MBE/Willmott's d against station data. Deterioration: accuracy/weighted F1 on held-out/temporal test sets |
| Subsystem | Each subsystem against independent ground truth not used in calibration | Hold-out site or season; independent weather station; expert review of extracted rules |
| Integration | The coupled system's combined output | Retrospective comparison of predicted maintenance-priority rankings against actual subsequent maintenance records; face validity review with KALAM/MBJB stakeholders |
| Robustness | Sensitivity of conclusions to calibration uncertainty | Scenario analysis varying visitor-flow and microclimate parameters; rank-stability statistics (e.g., Kendall's tau) |

---

## Grounding and Source Materials

This framework is grounded in, and designed to extend, the following materials from the Johor Bahru heritage research programme: the microclimate maintenance-priority study and its companion XAI/decision-support paper (XGBoost, SHAP, comparative model table), the multi-site machine-learning maintenance study (MLBS) covering the six Johor Bahru heritage buildings, the YOLOv11n/ECA real-time human-detection study, the Space Syntax multi-scale accessibility methodology (Integration/NACH), the depthmapX/OSM workflow already built for the Johor Bahru heritage core, and the existing SSS15 keynote synthesis of the COE/RG research ecosystem.
