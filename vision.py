import cv2
import numpy as np
import pyautogui

def find_ice_crystal(debug=False):
    # Take screenshot
    screenshot = pyautogui.screenshot()
    frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    h, w = frame.shape[:2]

    # ROI to ignore sky & UI
    roi = frame[int(h*0.4):int(h*0.9), int(w*0.15):int(w*0.85)]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # UPDATED: Much wider range for ice crystals with different lighting
    # Handles dark caves, bright areas, and various blue/cyan shades
    lower_ice = np.array([75, 20, 100])    # Lower hue, very low saturation, lower brightness
    upper_ice = np.array([135, 255, 255])  # Wider hue range to catch all blues/cyans
    color_mask = cv2.inRange(hsv, lower_ice, upper_ice)

    # Edge detection for spikes (more sensitive for darker crystals)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 120)  # Lower thresholds to catch darker edges

    combined = cv2.bitwise_and(color_mask, edges)

    # Morphology to clean up noise
    kernel = np.ones((5, 5), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    # Find ice contours
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    debug_frame = frame.copy() if debug else None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # UPDATED: Wider area range for different crystal sizes and distances
        if area < 200 or area > 20000:  # Lower minimum, higher maximum
            continue

        x, y, w2, h2 = cv2.boundingRect(cnt)
        center_x = int(w*0.15) + x + w2 // 2
        center_y = int(h*0.4) + y + h2 // 2

        # Look ABOVE the crystal for white text (HP indicator)
        # The "7040 HP" text appears MUCH higher above the crystal
        hb_top = max(0, int(h*0.4) + y - int(h2*3.5))  # MUCH higher - 3.5x crystal height
        hb_bottom = int(h*0.4) + y + 10  # Include top of crystal
        hb_left = max(0, int(w*0.15) + x - int(w2*0.5))  # Wider
        hb_right = min(w, int(w*0.15) + x + int(w2*1.5))  # Wider

        healthbar_region = frame[hb_top:hb_bottom, hb_left:hb_right]

        is_mineable = False
        if healthbar_region.size > 0:
            # Convert to grayscale and look for WHITE text (HP numbers)
            gray_hb = cv2.cvtColor(healthbar_region, cv2.COLOR_BGR2GRAY)
            
            # ADAPTIVE threshold - works better in varying lighting
            _, white_mask = cv2.threshold(gray_hb, 160, 255, cv2.THRESH_BINARY)  # Even lower for dark caves
            
            # Count white pixels
            white_pixels = cv2.countNonZero(white_mask)
            
            # More strict detection - need BOTH white pixels AND proper distribution
            if white_pixels > 120:  # Slightly lower for darker text in caves
                # Check if white pixels are concentrated (text-like) not scattered
                contours_white, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Look for text-like regions (compact clusters)
                has_text = False
                for wc in contours_white:
                    w_area = cv2.contourArea(wc)
                    if w_area > 15:  # Slightly lower threshold
                        has_text = True
                        break
                
                if has_text:
                    is_mineable = True
                
                if debug:
                    print(f"  → White pixels: {white_pixels}, has_text: {has_text}")
            
            # ALTERNATIVE: Look for red health bar underneath the text
            hsv_hb = cv2.cvtColor(healthbar_region, cv2.COLOR_BGR2HSV)
            
            # Red HSV ranges (red wraps around) - slightly more lenient
            red_mask1 = cv2.inRange(hsv_hb, np.array([0, 80, 80]), np.array([10, 255, 255]))
            red_mask2 = cv2.inRange(hsv_hb, np.array([170, 80, 80]), np.array([180, 255, 255]))
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            
            red_pixels = cv2.countNonZero(red_mask)
            if red_pixels > 60:  # Balanced threshold
                is_mineable = True
                if debug:
                    print(f"  → Red pixels (health bar) found: {red_pixels}")

        if debug and debug_frame is not None:
            # Draw ice crystal box
            color = (0, 255, 0) if is_mineable else (0, 0, 255)
            cv2.rectangle(debug_frame, 
                         (int(w*0.15)+x, int(h*0.4)+y), 
                         (int(w*0.15)+x+w2, int(h*0.4)+y+h2), 
                         color, 2)
            # Draw HP search area
            cv2.rectangle(debug_frame, 
                         (hb_left, hb_top), 
                         (hb_right, hb_bottom), 
                         (255, 255, 0), 1)
            # Label
            label = "MINEABLE" if is_mineable else "DECO"
            cv2.putText(debug_frame, label, 
                       (int(w*0.15)+x, int(h*0.4)+y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        results.append({
            "position": (center_x, center_y),
            "area": area,
            "mineable": is_mineable,
            "rect": (int(w*0.15)+x, int(h*0.4)+y, w2, h2)
        })

    if debug and debug_frame is not None:
        # Only save main debug output, not individual regions
        cv2.imwrite("debug_output.png", debug_frame)
        print(f"Debug image saved as 'debug_output.png'")

    return results