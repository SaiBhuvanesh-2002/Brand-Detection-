import cv2
import numpy as np
from pathlib import Path

def compare_refs():
    current_dir = Path(__file__).parent
    ref_main = current_dir / "reference_main.jpg"
    ref_cam2 = current_dir / "reference_cam2.jpg"
    
    if not ref_main.exists() or not ref_cam2.exists():
        print("Missing reference files.")
        return

    img1 = cv2.imread(str(ref_main))
    img2 = cv2.imread(str(ref_cam2))

    # Calculate histograms
    hist1 = cv2.calcHist([cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)], [0], None, [50], [0, 180])
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)

    hist2 = cv2.calcHist([cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)], [0], None, [50], [0, 180])
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)

    # Compare them directly
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    print(f"Similarity between Main and Cam2 References: {similarity:.4f}")
    
    if similarity > 0.8:
        print("WARNING: References are too similar! Histograms cannot distinguish them easily.")
        print("Recommendation: Use a different reference frame for Cam2 with distinct colors.")

if __name__ == "__main__":
    compare_refs()
