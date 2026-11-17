import os
import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

def extract_features(image_path):
    """Extracts high-precision geometric and photometric features from a hole crop."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
        
    # Binarize the crop to isolate the hole
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
        
    # --- NEW LOGIC: FIND HOLE CLOSEST TO CENTER ---
    center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
    best_cnt = None
    min_dist = float('inf')
    
    for cnt in contours:
        if cv2.contourArea(cnt) < 50:  # Filter out tiny dust/noise pixels
            continue
            
        M = cv2.moments(cnt)
        if M["m00"] == 0: 
            continue
            
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Calculate distance from this blob to the dead-center of the crop
        dist = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
        
        if dist < min_dist:
            min_dist = dist
            best_cnt = cnt
            
    if best_cnt is None:
        return None
    # ----------------------------------------------
        
    area = cv2.contourArea(best_cnt)
    perimeter = cv2.arcLength(best_cnt, True)
    
    if perimeter == 0:
        return None
        
    # 1. Circularity (1.0 is perfect circle. Helps find radii defects)
    circularity = 4 * np.pi * (area / (perimeter * perimeter))
    
    # 2. Position Offset from Center (Helps find positional drift)
    M = cv2.moments(best_cnt)
    if M["m00"] == 0: return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    expected_cx, expected_cy = img.shape[1]//2, img.shape[0]//2
    offset_dist = np.sqrt((cx - expected_cx)**2 + (cy - expected_cy)**2)
    
    # 3. Mean Intensity (Is it actually a dark drilled hole?)
    mask = np.zeros_like(img)
    cv2.drawContours(mask, [best_cnt], -1, 255, -1)
    mean_intensity = cv2.mean(img, mask=mask)[0]
    
    return {
        "file": os.path.basename(image_path),
        "area": area,
        "circularity": circularity,
        "offset_dist": offset_dist,
        "mean_intensity": mean_intensity
    }

def train_and_classify():
    print("="*50)
    print(" AVID ML Sub-System: High-Precision Categorization")
    print("="*50)
    
    # Points to your newly cleaned up run folder
    dataset_dir = "reports/real_run/ml_dataset" 
    if not os.path.exists(dataset_dir):
        print(f"[ERROR] Dataset not found at {dataset_dir}. Run the pipeline first!")
        return
        
    print("[INFO] Extracting high-precision features from crops...")
    data = []
    for filename in os.listdir(dataset_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            features = extract_features(os.path.join(dataset_dir, filename))
            if features:
                data.append(features)
                
    df = pd.DataFrame(data)
    
    # Define our ML Features
    X = df[["area", "circularity", "offset_dist", "mean_intensity"]]
    
    print("[INFO] Training Machine Learning Model (Isolation Forest)...")
    # Isolation forest learns the "perfect" hole and flags deviations
    model = IsolationForest(contamination=0.15, random_state=42)
    df['ml_anomaly_score'] = model.fit_predict(X)
    
    # --- NEW SUB-CLASS CLASSIFIER LOGIC ---
    def classify_defect_type(row):
        if row['ml_anomaly_score'] == -1:
            if row['mean_intensity'] > 45:
                return 'BRIGHT_OBSTRUCTION'       # U6-type
            elif row['circularity'] < 0.08:
                return 'BURR_OR_TEAROUT'          # T11,T12,T6,T18-type
            else:
                return 'SURFACE_ANOMALY_ML_ONLY'  # T3-type
        return 'NOMINAL_AEROSPACE_GRADE'
        
    df['ml_category'] = df.apply(classify_defect_type, axis=1)
    # --------------------------------------
    
    # Save the intelligent report directly into your reports folder
    output_path = "reports/ml_classification_report.csv"
    df.to_csv(output_path, index=False)
    
    print("\n--- ML Categorization Results ---")
    # Updated the print statement to show intensity as well, since it's a key metric now
    print(df[['file', 'ml_category', 'circularity', 'mean_intensity']].head(15))
    print(f"\n[SUCCESS] Analyzed {len(df)} holes. Full intel saved to '{output_path}'")

    # --- NEW: LAUNCH DASHBOARD AUTOMATICALLY ---
    print("\n[INFO] Booting up AVID Intelligence Dashboard...")
    import subprocess
    subprocess.Popen(["python", "avid_dashboard.py"])
    # -------------------------------------------

if __name__ == "__main__":
    train_and_classify()