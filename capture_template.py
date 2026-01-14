import cv2
import pyautogui
import numpy as np
import time

print("=" * 60)
print("HEALTH BAR TEMPLATE CAPTURE TOOL")
print("=" * 60)
print()
print("Instructions:")
print("1. Position your mouse ABOVE a mineable crystal's health bar")
print("2. Press SPACE to start capture countdown (3 seconds)")
print("3. Script will capture a small region around your mouse")
print("4. The captured image will be saved as 'healthbar_template.png'")
print()
print("Tips:")
print("- Make sure the health bar is clearly visible")
print("- Try to capture just the red bar + white text")
print("- You can run this multiple times to get the best template")
print()
print("Press SPACE when ready, or ESC to quit")
print("=" * 60)

import keyboard

while True:
    if keyboard.is_pressed('space'):
        print("\n⏳ Capturing in 3 seconds... Position your mouse!")
        time.sleep(3)
        
        # Get mouse position
        mouse_x, mouse_y = pyautogui.position()
        print(f"📍 Mouse position: ({mouse_x}, {mouse_y})")
        
        # Take screenshot
        screenshot = pyautogui.screenshot()
        frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        
        # Capture region around mouse (adjust size as needed)
        width, height = 150, 40  # Adjust these if needed
        left = max(0, mouse_x - width // 2)
        top = max(0, mouse_y - height // 2)
        right = min(frame.shape[1], left + width)
        bottom = min(frame.shape[0], top + height)
        
        # Extract template
        template = frame[top:bottom, left:right]
        
        # Save to assets folder
        import os
        os.makedirs("assets", exist_ok=True)
        cv2.imwrite("assets/healthbar_template.png", template)
        
        print(f"✅ Template saved as 'assets/healthbar_template.png'")
        print(f"   Size: {template.shape[1]}x{template.shape[0]} pixels")
        print()
        print("Preview saved as 'template_preview.png' for verification")
        
        # Save a preview with border
        preview = cv2.copyMakeBorder(template, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 255, 0])
        cv2.imwrite("template_preview.png", preview)
        
        print()
        print("✓ Done! You can now run your main detection script.")
        print("  If the template doesn't work well, run this again for a better capture.")
        break
    
    if keyboard.is_pressed('esc'):
        print("\n❌ Cancelled")
        break
    
    time.sleep(0.1)