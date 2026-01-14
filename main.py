import time
import keyboard
from vision import find_ice_crystal

print("Starting ice crystal detection...")
print("Press F8 to stop")
print("=" * 50)

iteration = 0
while True:
    if keyboard.is_pressed("f8"):
        print("\nStopping...")
        break

    iteration += 1
    print(f"\n[Scan #{iteration}]")
    
    # Debug mode always on - overwrites debug_output.png each scan
    crystals = find_ice_crystal(debug=True)
    
    if not crystals:
        print("  No ice crystals detected")
    else:
        for c in crystals:
            x, y = c["position"]
            status = "MINEABLE" if c["mineable"] else "decoration"
            print(f"  [FOUND] Ice crystal at ({x},{y}) → {status} (area={c['area']})")

    time.sleep(1)