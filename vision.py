import cv2
import numpy as np
import pyautogui
import os

def find_ice_crystal(debug=False):
    # Take screenshot
    screenshot = pyautogui.screenshot()
    frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    h, w = frame.shape[:2]

    # ROI - EXPANDED to capture more crystals
    roi = frame[int(h*0.2):int(h*0.95), int(w*0.05):int(w*0.95)]
    roi_offset_y = int(h*0.2)
    roi_offset_x = int(w*0.05)
    
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast for dark caves
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Find bright areas (crystals are brighter than surroundings)
    _, bright_mask = cv2.threshold(enhanced, 120, 255, cv2.THRESH_BINARY)  # Increased from 100 to 120
    
    # Strong edge detection
    edges = cv2.Canny(enhanced, 60, 150)  # Increased thresholds for sharper edges only
    
    # Dilate edges to make them thicker
    kernel_edge = np.ones((3, 3), np.uint8)
    edges_thick = cv2.dilate(edges, kernel_edge, iterations=2)
    
    # Combine: areas that are BOTH bright AND have edges nearby
    combined = cv2.bitwise_and(bright_mask, edges_thick)
    
    # Morphology to connect crystal fragments and remove noise
    kernel = np.ones((5, 5), np.uint8)  # Smaller kernel
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)  # Less aggressive
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)  # Remove small noise

    # Find contours
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if debug:
        print(f"  Found {len(contours)} potential contours before filtering")

    results = []
    debug_frame = frame.copy() if debug else None

    # Load health bar template if exists
    healthbar_template = None
    template_path = "assets/healthbar_template.png"
    if os.path.exists(template_path):
        healthbar_template = cv2.imread(template_path)
        if debug:
            print(f"✓ Loaded health bar template")
    elif debug:
        print(f"⚠ No health bar template - using pixel detection")

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Stricter area range
        if area < 400 or area > 12000:  # Tighter range
            continue
        
        x, y, w2, h2 = cv2.boundingRect(cnt)
        
        # Stricter aspect ratio filter - crystals are relatively compact
        aspect_ratio = w2 / float(h2)
        if aspect_ratio > 2.5 or aspect_ratio < 0.3:  # More strict
            continue
        
        # Check solidity - reject hollow/irregular shapes
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area
            if solidity < 0.4:  # Reject very hollow shapes
                continue

        center_x = roi_offset_x + x + w2 // 2
        center_y = roi_offset_y + y + h2 // 2

        # Health bar search region
        hb_top = max(0, roi_offset_y + y - int(h2*3.5))
        hb_bottom = roi_offset_y + y + 10
        hb_left = max(0, roi_offset_x + x - int(w2*0.5))
        hb_right = min(w, roi_offset_x + x + int(w2*1.5))

        healthbar_region = frame[hb_top:hb_bottom, hb_left:hb_right]

        is_mineable = False
        detection_method = "none"
        best_match = 0

        if healthbar_region.size > 0:
            # METHOD 1: Template Matching
            if healthbar_template is not None:
                scales = [0.8, 1.0, 1.2, 1.4]
                for scale in scales:
                    template_resized = cv2.resize(healthbar_template, None, fx=scale, fy=scale)
                    th, tw = template_resized.shape[:2]
                    if th > healthbar_region.shape[0] or tw > healthbar_region.shape[1]:
                        continue
                    result = cv2.matchTemplate(healthbar_region, template_resized, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(result)
                    if max_val > best_match:
                        best_match = max_val
                
                if best_match > 0.65:
                    is_mineable = True
                    detection_method = f"template({best_match:.2f})"
            
            # METHOD 2: Pixel detection fallback
            if not is_mineable:
                gray_hb = cv2.cvtColor(healthbar_region, cv2.COLOR_BGR2GRAY)
                _, white_mask = cv2.threshold(gray_hb, 160, 255, cv2.THRESH_BINARY)
                white_pixels = cv2.countNonZero(white_mask)
                
                if white_pixels > 150:
                    contours_white, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for wc in contours_white:
                        if cv2.contourArea(wc) > 20:
                            is_mineable = True
                            detection_method = f"white_text({white_pixels}px)"
                            break
                
                if not is_mineable:
                    hsv_hb = cv2.cvtColor(healthbar_region, cv2.COLOR_BGR2HSV)
                    red_mask1 = cv2.inRange(hsv_hb, np.array([0, 80, 80]), np.array([10, 255, 255]))
                    red_mask2 = cv2.inRange(hsv_hb, np.array([170, 80, 80]), np.array([180, 255, 255]))
                    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
                    red_pixels = cv2.countNonZero(red_mask)
                    
                    if red_pixels > 80:
                        is_mineable = True
                        detection_method = f"red_bar({red_pixels}px)"

        if debug and debug_frame is not None:
            color = (0, 255, 0) if is_mineable else (0, 0, 255)
            cv2.rectangle(debug_frame, 
                         (roi_offset_x+x, roi_offset_y+y), 
                         (roi_offset_x+x+w2, roi_offset_y+y+h2), 
                         color, 2)
            cv2.rectangle(debug_frame, 
                         (hb_left, hb_top), 
                         (hb_right, hb_bottom), 
                         (255, 255, 0), 1)
            label = f"MINEABLE[{detection_method}]" if is_mineable else "DECO"
            cv2.putText(debug_frame, label, 
                       (roi_offset_x+x, roi_offset_y+y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        results.append({
            "position": (center_x, center_y),
            "area": area,
            "mineable": is_mineable,
            "detection_method": detection_method,
            "rect": (roi_offset_x+x, roi_offset_y+y, w2, h2)
        })

    if debug and debug_frame is not None:
        # Draw ROI
        cv2.rectangle(debug_frame, 
                     (roi_offset_x, roi_offset_y), 
                     (int(w*0.95), int(h*0.95)), 
                     (0, 255, 255), 2)
        
        # Save debug images
        cv2.imwrite("debug_output.png", debug_frame)
        cv2.imwrite("debug_1_enhanced.png", enhanced)
        cv2.imwrite("debug_2_bright_mask.png", bright_mask)
        cv2.imwrite("debug_3_edges.png", edges_thick)
        cv2.imwrite("debug_4_combined.png", combined)
        print(f"Debug images saved - new approach using brightness + edges")

    return results