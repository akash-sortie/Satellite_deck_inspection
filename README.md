# AVID: Automated Vision Inspection for Decks

**AVID** (Automated Vision Inspection for Decks) is a hybrid Machine Vision and Unsupervised Machine Learning pipeline engineered for sub-millimeter aerospace quality control. 

Designed to inspect 43 micro-features on high-resolution metallic decks, AVID dynamically corrects severe radial lens distortion and uses mathematical feature extraction to categorize manufacturing anomalies with extreme precision.

---

## 🏗️ Core Architecture

The system abandons standard Deep Learning (which fails under high compression and data starvation) in favor of a bespoke, two-phase deterministic pipeline:

### 1. Phase 1: Macro-Localization (Computer Vision)
* **Pre-Processing:** Neutralizes harsh metallic surface glare using Contrast Limited Adaptive Histogram Equalization (CLAHE) and Bilateral Filtering.
* **Geometric Registration:** Calculates a precise Perspective Transformation (Homography matrix) using 4 hardcoded fiducial anchors to bridge the 2D CAD blueprint to the raw physical photograph.
* **Distortion Compensation:** Expands the geometric search radius to 20.0mm to capture outer holes shifted by severe (16.5mm) radial lens distortion, extracting 150x150 pixel Regions of Interest (ROIs).

### 2. Phase 2: Micro-Inspection (Machine Learning)
* **Center-Weighted Extraction:** Algorithmically ignores neighboring features, extracting sub-pixel geometry (Area, Circularity, Position Offset, Mean Intensity) strictly from the target hole.
* **Unsupervised Anomaly Detection:** An **Isolation Forest** model learns the mathematical baseline of a "perfect" aerospace hole from a small dataset, flagging any deviations without requiring thousands of training images.
* **Sub-Class Diagnostics:** Categorizes anomalies into actionable physical defects rather than generic failures.

### 3. Phase 3: The Command Center
A fully integrated, dark-themed PyQt6 dashboard providing real-time execution control and embedded Matplotlib analytics to visualize the ML isolation clusters.

---

## 📊 Defect Categorization

The ML pipeline categorizes every hole into one of four states:
* 🟢 **NOMINAL_AEROSPACE_GRADE:** Perfect circularity and intensity.
* 🔵 **BRIGHT_OBSTRUCTION:** Intensity > 45. The hole is completely washed out by a massive glare spike.
* 🔴 **BURR_OR_TEAROUT:** Circularity < 0.08. Physical metal around the hole is warped, scratched, or improperly drilled.
* 🟠 **SURFACE_ANOMALY_ML_ONLY:** The Isolation Forest detected a subtle deviation (e.g., micro-scratch inside the crop).

---

## 📁 Repository Structure

```text
AVID_Project/
├── data_image.jpeg                # Raw input photograph
├── real_deck.csv                  # Blueprint coordinates
├── requirements.txt               # Environment dependencies
├── avid_dashboard_pro.py          # PyQt6 Command Center UI
├── pixel_pick.py                  # Manual fiducial anchor utility
├── AVID/
│   ├── pipeline.py                # Phase 1: Macro-Vision CV script
│   ├── ml_classifier.py           # Phase 2: Micro-Inspection ML script
│   └── core/
│       ├── cad_parser.py          
│       ├── hole_detector.py       
│       └── image_preprocessor.py  
└── reports/
    ├── ml_classification_report.csv # Final ML intelligence
    └── real_run/
        ├── inspection_overlay.jpg   # Visual macro-vision map
        └── ml_dataset/              # Extracted 150x150 ROI crops



🚀 Installation & Setup
This software is designed to run entirely offline in secure, air-gapped intranet environments.

Clone or extract the repository to your local drive.

Install the required dependencies:
    pip install -r requirements.txt

💻 Usage Instructions
The entire system is controlled via the AVID Command Center.

Launch the Dashboard:
    python avid_dashboard.py

Execute Phase 1: Click INITIALIZE MACRO-VISION to scan the image and generate the visual map.

Execute Phase 2: Click EXECUTE ML CLASSIFIER to run the Isolation Forest and grade the holes.

View Analytics: The dashboard will automatically update with the scatter plot and data grid once the ML finishes running.

Adding New Camera Data:
To run the system on a new image, replace data_image.jpeg in the root folder, run python pixel_pick.py to get the 4 new anchor points, update AVID/homography.py, and run the dashboard.