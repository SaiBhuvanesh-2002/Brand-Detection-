import cv2
import logging
import csv
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO
import numpy as np 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class LogoSponsorshipDetector:
    def __init__(self, model_path, video_path, output_path):
        self.video_path = Path(video_path)
        self.output_path = Path(output_path)
        
        # Load YOLOv11 Model
        logging.info(f"Loading model from {model_path}...")
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise e
        
        # Camera Re-ID settings
        self.camera_fingerprints = {} 
        self.reid_threshold = 0.70 # Lowered to 0.70 to assume more variance
        self.reference_hists = {} # Stores {camera_id: histogram}

        # Financial settings (Default)
        self.base_rate_30s = 100.0 # $100 per 30 seconds

    def _calculate_valuation(self, duration_sec, quality_score):
        # Value = (Duration / 30s) * Base_Rate * (Quality_Score / 100)
        return (duration_sec / 30.0) * self.base_rate_30s * (quality_score / 100.0)

    def _format_time(self, frame_idx, fps):
        seconds = frame_idx / fps
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{int(h):02d}:{int(m):02d}:{s:.2f}"

    def process(self):
        if not self.video_path.exists():
            logging.error(f"Video not found: {self.video_path}")
            return

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
             logging.error(f"Could not open video file.")
             return

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_area = width * height
        
        # Output setup
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(self.output_path), fourcc, fps, (width, height))
        
        # Analytics state
        frame_idx = 0
        
        # --- Load Reference Histograms (All Angles) ---
        # Resolve path relative to this script
        current_dir = Path(__file__).parent
        ref_files = list(current_dir.glob("reference_*.jpg"))
        
        if not ref_files:
            logging.error(f"No reference images found in {current_dir}! Run extract_reference.py first.")
            return

        for ref_path in ref_files:
            # Extract Camera ID from filename: reference_cam2.jpg -> cam2
            cam_id = ref_path.stem.replace("reference_", "")
            
            ref_img = cv2.imread(str(ref_path))
            if ref_img is None:
                logging.warning(f"Could not read {ref_path}. Skipping.")
                continue
                
            ref_hist = cv2.calcHist([cv2.cvtColor(ref_img, cv2.COLOR_BGR2HSV)], [0], None, [50], [0, 180])
            cv2.normalize(ref_hist, ref_hist, 0, 1, cv2.NORM_MINMAX)
            self.reference_hists[cam_id] = ref_hist
            logging.info(f"Loaded Reference Angle: {cam_id}")
            
        logging.info(f"Total Reference Angles Loaded: {len(self.reference_hists)}")
        
        active_camera_id = None
        frames_since_active = 0 
        
        import numpy as np # Ensure numpy is imported

        # --- Metrics Helper (Mapped to Brief) ---
        def calculate_9_variables(box, frame, frame_area, frame_center, track_conf=0.9):
            x1, y1, x2, y2 = map(int, box)
            
            # 1. Size/Scale (Score 0-100)
            box_area = (x2-x1) * (y2-y1)
            share_pct = (box_area / frame_area) * 100
            size_score = min(share_pct * 5, 100)
            
            # 2. Legibility (Clarity) (0-100)
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                clarity_score = 0
                clutter_score = 0
            else:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                variance = cv2.Laplacian(gray, cv2.CV_64F).var()
                clarity_score = min(max(variance - 50, 0) / 5, 100)
                
                # 4. Clutter (100 - ROI Edge Density) (Heuristic)
                edges = cv2.Canny(gray, 100, 200)
                edge_density = np.count_nonzero(edges) / edges.size
                clutter_score = max(0, 100 - (edge_density * 400)) # Heuristic

            # 6. Position (Centrality) (0-100)
            box_cx, box_cy = (x1+x2)/2, (y1+y2)/2
            dist_x = abs(box_cx - frame_center[0])
            dist_y = abs(box_cy - frame_center[1])
            max_dist = np.sqrt(frame_center[0]**2 + frame_center[1]**2)
            curr_dist = np.sqrt(dist_x**2 + dist_y**2)
            position_score = max(0, (1 - (curr_dist / max_dist)) * 100)
            
            # 8. Integration (Heuristic based on position)
            # Center = Natural (100), Edge = Overlay (80)? 
            # Simplified: Matches Position Score for now.
            integration_score = position_score 

            # 9. Obstruction (Confidence-based)
            # High confidence = Low obstruction
            obstruction_score = min(track_conf * 100, 100)
            
            return {
                "Size/Scale": size_score,
                "Legibility": clarity_score,
                "Position": position_score,
                "Clutter": clutter_score,
                "Integration": integration_score,
                "Obstruction": obstruction_score,
                "Share of Screen": share_pct
            }

        def calculate_iou(boxA, boxB):
            # determine the (x, y)-coordinates of the intersection rectangle
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])

            # compute the area of intersection rectangle
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

            # compute the area of both the prediction and ground-truth rectangles
            boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
            boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

            # compute the intersection over union by taking the intersection
            # area and dividing it by the sum of prediction + ground-truth
            # areas - the interesection area
            iou = interArea / float(boxAArea + boxBArea - interArea)
            return iou

        frame_center = (width // 2, height // 2)

        # Track Data Structure
        tracks = defaultdict(lambda: {
            'start': 0, 'end': 0, 'camera': 0, 'counted': False,
            'frames': 0,
            
            # 9 Variables Accumulators
            'sum_size': 0.0, 'sum_legibility': 0.0, 'sum_position': 0.0,
            'sum_clutter': 0.0, 'sum_integration': 0.0, 'sum_obstruction': 0.0,
            
            # Stability (Camera Movement)
            'sum_iou': 0.0, 'iou_frames': 0, 'last_box': None,
            
            # Screen Share
            'max_share': 0.0, 'sum_share': 0.0
        })
        impressions = defaultdict(int)

        logging.info(f"Starting analysis on {self.video_path.name}...")
        
        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success: break
                frame_idx += 1
                
                # --- 1. Scene Fingerprinting (Match against all refs) ---
                curr_hist = cv2.calcHist([cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)], [0], None, [50], [0, 180])
                cv2.normalize(curr_hist, curr_hist, 0, 1, cv2.NORM_MINMAX)
                
                best_match_score = 0.0
                best_match_id = None
                
                # Compare against all references
                scores = {}
                for cam_id, ref_hist in self.reference_hists.items():
                    score = cv2.compareHist(ref_hist, curr_hist, cv2.HISTCMP_CORREL)
                    scores[cam_id] = score
                    if score > best_match_score:
                        best_match_score = score
                        best_match_id = cam_id
                
                # --- Hysteresis / Sticky Logic ---
                # If we have an active camera, only switch if the new best match is SIGNIFICANTLY better
                # This prevents "flickering" if Cam 1 and Cam 2 are very similar.
                if active_camera_id is not None and best_match_id != active_camera_id:
                     current_cam_score = scores.get(active_camera_id, 0)
                     # Margin: New cam must be 0.05 (5%) better than current cam to switch
                     if best_match_score < (current_cam_score + 0.05):
                         best_match_id = active_camera_id
                         best_match_score = current_cam_score

                # Check threshold
                is_active_frame = (best_match_score > self.reid_threshold)

                if is_active_frame:
                    frames_since_active = 0
                    active_camera_id = best_match_id
                    is_active = True
                else:
                    frames_since_active += 1
                    if frames_since_active < 10: # 10-frame buffer to handle blips
                        is_active = True
                        # Keep previous camera ID active during buffer
                    else:
                        is_active = False
                        active_camera_id = None

                if is_active and active_camera_id:
                    # --- 2. YOLO Detection (Active) ---
                    results = self.model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")
                    
                    if results[0].boxes.id is not None:
                        ids = results[0].boxes.id.cpu().numpy().astype(int)
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        confs = results[0].boxes.conf.cpu().numpy()
                        
                        for track_id, box, conf in zip(ids, boxes, confs):
                            if tracks[track_id]['frames'] == 0:
                                tracks[track_id]['start'] = frame_idx
                                tracks[track_id]['camera'] = active_camera_id # Assign detected Camera ID
                            
                            # 3. Camera Movement (Stability)
                            if tracks[track_id]['last_box'] is not None:
                                 iou = calculate_iou(tracks[track_id]['last_box'], box)
                                 tracks[track_id]['sum_iou'] += iou
                                 tracks[track_id]['iou_frames'] += 1
                            tracks[track_id]['last_box'] = box
                            tracks[track_id]['end'] = frame_idx
                            
                            # 4. Calculate 9 Variables
                            metrics = calculate_9_variables(box, frame, frame_area, frame_center, conf)
                            
                            # Accumulate
                            tracks[track_id]['sum_size'] += metrics['Size/Scale']
                            tracks[track_id]['sum_legibility'] += metrics['Legibility']
                            tracks[track_id]['sum_position'] += metrics['Position']
                            tracks[track_id]['sum_clutter'] += metrics['Clutter']
                            tracks[track_id]['sum_integration'] += metrics['Integration']
                            tracks[track_id]['sum_obstruction'] += metrics['Obstruction']
                            
                            tracks[track_id]['max_share'] = max(tracks[track_id]['max_share'], metrics['Share of Screen'])
                            tracks[track_id]['sum_share'] += metrics['Share of Screen']
                            tracks[track_id]['frames'] += 1

                            # Duration Check (0.8s)
                            duration_sec = (tracks[track_id]['end'] - tracks[track_id]['start']) / fps
                            if duration_sec > 0.8 and not tracks[track_id]['counted']:
                                tracks[track_id]['counted'] = True
                                impressions[active_camera_id] += 1

                            # Visualization
                            color = (0, 255, 0) if tracks[track_id]['counted'] else (0, 165, 255)
                            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                            cv2.putText(frame, f"ID:{track_id}", (int(box[0]), int(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                # Draw Screen Info
                # Draw Screen Info
                status_color = (0, 255, 0) if is_active else (0, 0, 255)
                status_text = f"ACTIVE: {active_camera_id}" if is_active else "IDLE"
                
                # Debug info on screen
                debug_y = 90
                for cam, score in scores.items():
                     color = (0,255,0) if cam == active_camera_id else (200,200,200)
                     cv2.putText(frame, f"{cam}: {score:.3f}", (20, debug_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                     debug_y += 30

                cv2.putText(frame, f"{status_text}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                out.write(frame)

        except KeyboardInterrupt:
            logging.info("Interrupted. Saving...")
        
        finally:
            cap.release()
            out.release()
            
            # --- Generate Deliverables ---
            csv_path = self.output_path.parent / "brand_exposure_report.csv"
            
            total_value = 0.0
            total_duration = 0.0
            
            with open(csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                # Matches Project Brief Pillars
                writer.writerow([
                    "Impression ID", "Camera ID", "Start", "End", "Duration (s)", 
                    "Media Value ($)", "Qualitative Score", # Financial & Score
                    "Size/Scale", "Legibility", "Camera Movement (Stability)", # 9 Variables...
                    "Position", "Clutter", "Integration", "Obstruction", "Share of Screen %"
                ])
                
                imp_id = 1
                for tid, data in sorted(tracks.items(), key=lambda x: x[1]['start']):
                    if data['counted']:
                        frames = data['frames']
                        dur = (data['end'] - data['start']) / fps
                        
                        # Averaging Metrics
                        avg_size = data['sum_size'] / frames
                        avg_leg = data['sum_legibility'] / frames
                        avg_pos = data['sum_position'] / frames
                        avg_clutter = data['sum_clutter'] / frames
                        avg_int = data['sum_integration'] / frames
                        avg_obs = data['sum_obstruction'] / frames
                        avg_share = data['sum_share'] / frames
                        
                        # Stability (Camera Movement) Score (0-100)
                        # IoU 1.0 -> 100 Score. IoU 0.5 -> 50 Score.
                        avg_iou = data['sum_iou'] / data['iou_frames'] if data['iou_frames'] > 0 else 0
                        movement_score = avg_iou * 100
                        
                        # Final Qualitative Score (The formula)
                        # Brief: "9-variable score". We'll average the 7 normalized scores we have.
                        q_score = (avg_size + avg_leg + avg_pos + avg_clutter + avg_int + avg_obs + movement_score) / 7
                        
                        # Financial Valuation
                        val = self._calculate_valuation(dur, q_score)
                        
                        total_value += val
                        total_duration += dur
                        
                        writer.writerow([
                            imp_id, f"Camera {data['camera']}", 
                            self._format_time(data['start'], fps), self._format_time(data['end'], fps),
                            f"{dur:.2f}", 
                            f"${val:.2f}", f"{q_score:.1f}", # Value & Score
                            f"{avg_size:.1f}", f"{avg_leg:.1f}", f"{movement_score:.1f}",
                            f"{avg_pos:.1f}", f"{avg_clutter:.1f}", f"{avg_int:.1f}", f"{avg_obs:.1f}", 
                            f"{avg_share:.2f}%"
                        ])
                        imp_id += 1
            
            logging.info(f"Detailed Report saved to {csv_path}")

            # --- Structured Report (Text) ---
            txt_path = self.output_path.parent / "brand_exposure_summary.txt"
            with open(txt_path, "w") as f:
                f.write("BRAND EXPOSURE ASSESSMENT: GOA\n")
                f.write("=================================\n\n")
                f.write("1. EXECUTIVE SUMMARY\n")
                f.write(f"- Total Impressions:    {imp_id - 1}\n")
                f.write(f"- Total Airtime:        {total_duration:.2f} seconds\n")
                f.write(f"- Total Media Value:    ${total_value:,.2f}\n")
                f.write(f"- Avg Qualitative Score:{total_value/imp_id if imp_id > 1 else 0:.1f} / 100\n\n")
                
                f.write("2. CAMERA BREAKDOWN (ROI)\n")
                f.write(f"Top performing cameras detailed in {csv_path.name}\n")
            
            print(f"\nFinal Analysis Complete.")
            print(f"Total Value: ${total_value:,.2f}")
            print(f"Report: {txt_path}\n")

if __name__ == "__main__":
    current_dir = Path(__file__).parent
    video_file = current_dir / "input_video.mp4"
    model_file = current_dir / "best.pt"
    
    if not model_file.exists():
        logging.warning("Custom model not found. Using yolo11n.pt.")
        model_file = "yolo11n.pt" 
    else:
        logging.info(f"Using custom model: {model_file.name}")

    detector = LogoSponsorshipDetector(model_file, video_file, current_dir / "output_annotated.mp4")
    detector.process()