"""
Simple dataset capture utility for IDS cameras using ids_peak.

Usage:
- Run the script. It will initialize the IDS camera.
- You will be prompted to choose between loading camera settings from a file or performing automatic calibration.
- loading from settings file delivers best results
- take image with space bar
- quit with 'q'

"""

import os
import sys
import time

try:
    import msvcrt
except Exception:
    msvcrt = None

from ids_peak import ids_peak
from ids_peak import ids_peak_ipl_extension
import ids_peak_ipl.ids_peak_ipl as ids_ipl
import cv2
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "dataset")
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "camsettings.cset")


def next_image_index(folder: str, prefix: str = "image_") -> int:
    os.makedirs(folder, exist_ok=True)
    existing = [f for f in os.listdir(folder) if f.startswith(prefix)]
    nums = []
    for f in existing:
        name, _ = os.path.splitext(f)
        try:
            n = int(name.replace(prefix, ""))
            nums.append(n)
        except Exception:
            continue
    return max(nums) + 1 if nums else 1


def wait_for_key() -> str:
    """Wait for a single key press and return the key as string. Windows only (msvcrt).
    Returns lowercase character, space as ' '."""
    if msvcrt is None:
        # Fallback to input() if msvcrt is not available (non-Windows)
        ch = input().strip()
        return ch.lower()
    while True:
        if msvcrt.kbhit():
            key = msvcrt.getch()
            # arrow keys and special return two bytes; ignore them
            if key in (b"\x00", b"\xe0"):
                msvcrt.getch()
                continue
            try:
                return key.decode("utf-8").lower()
            except Exception:
                return ""
def autocalibrate_camera(remote_nodemap):
    # Automatic calibration for your IDS U3-34Ex-C camera
            print("Starting automatic calibration...")
            print("Point camera at a well-lit scene with white/neutral colors for best results")
            
            # Your camera only has BlackLevelAuto, not ExposureAuto/GainAuto
            # Enable BlackLevelAuto temporarily to let camera adjust
            try:
                bl_node = remote_nodemap.FindNode('BlackLevelAuto')
                print(f"Current BlackLevelAuto: {bl_node.CurrentEntry().StringValue()}")
                
                # Enable continuous black level adjustment
                bl_node.SetCurrentEntry('ContinuousWithOffset')
                print('✓ Enabled BlackLevelAuto = ContinuousWithOffset')
                
                # Let the camera stabilize and adjust black levels
                print("Calibrating black level (waiting 3 seconds)...")
                time.sleep(3.0)
                
                # Lock it to Off to freeze the calibrated black level
                bl_node.SetCurrentEntry('Off')
                print('✓ Locked BlackLevelAuto = Off')
                
            except Exception as e:
                print(f'✗ BlackLevelAuto adjustment failed: {e}')
            
            # Read and display current exposure/gain settings
            # (These are manually set via SensorExposureTimeClocks and Gain nodes)
            print("\nCurrent camera settings:")
            try:
                exp_clocks = remote_nodemap.FindNode('SensorExposureTimeClocks').Value()
                print(f"  SensorExposureTimeClocks: {exp_clocks}")
            except Exception as e:
                print(f"  Could not read exposure: {e}")
            
            try:
                # Your camera has Gain with GainSelector
                gain_selector = remote_nodemap.FindNode('GainSelector')
                gain_selector.SetCurrentEntry('AnalogAll')
                gain_node = remote_nodemap.FindNode('Gain')
                current_gain = gain_node.Value()
                print(f"  Gain (AnalogAll): {current_gain:.4f}")
            except Exception as e:
                print(f"  Could not read gain: {e}")
            
            try:
                black_level = remote_nodemap.FindNode('BlackLevelInt').Value()
                print(f"  BlackLevelInt: {black_level}")
            except Exception as e:
                print(f"  Could not read black level: {e}")

def main():
    print("Initializing IDS library...")
    ids_peak.Library.Initialize()

    device_manager = ids_peak.DeviceManager.Instance()
    try:
        device_manager.Update()
        if device_manager.Devices().empty():
            print("No IDS device found. Connect camera and try again.")
            return 1

        # Open first available device (non-interactive)
        device = device_manager.Devices()[0].OpenDevice(ids_peak.DeviceAccessType_Control)

        # Remote nodemap and load default user set if available
        remote_nodemap = device.RemoteDevice().NodeMaps()[0]
        try:
            remote_nodemap.FindNode("UserSetSelector").SetCurrentEntry("Default")
            remote_nodemap.FindNode("UserSetLoad").Execute()
            remote_nodemap.FindNode("UserSetLoad").WaitUntilDone()

        except Exception:
            # If the camera doesn't expose these nodes, continue
            pass

        # Ask user whether to use config file or automatic calibration
        use_config = None
        while use_config is None:
            ans = input("Use camera settings file if available? (y = file, n = auto) [y/n]: ").strip().lower()
            if ans in ("y", "yes"):
                use_config = True
            elif ans in ("n", "no"):
                use_config = False
            if use_config is None:
                print("invalid input, please enter 'y' or 'n'.")

        # configuring camera
        if use_config:
            try:
                remote_nodemap.LoadFromFile(CONFIG_FILE)
                
            except Exception as e:
                str_error = str(e)
                print("Exception: " + str_error,"\n")
                print(f"Failed to load camera settings from {CONFIG_FILE}. Proceeding with automatic calibration.\n")
                autocalibrate_camera(remote_nodemap)
        else:
            autocalibrate_camera(remote_nodemap)

        # Open the first data stream
        data_stream = device.DataStreams()[0].OpenDataStream()

        # Payload size for buffer allocation
        payload_size = remote_nodemap.FindNode("PayloadSize").Value()

        # Minimum number of required buffers
        buffer_count = data_stream.NumBuffersAnnouncedMinRequired()
        print(f"Allocating and queueing {buffer_count} buffers...")

        # Allocate and announce buffers
        buffers = []
        for _ in range(buffer_count):
            buf = data_stream.AllocAndAnnounceBuffer(payload_size)
            buffers.append(buf)

        # Queue the buffers so the transport layer can fill them
        for buf in buffers:
            data_stream.QueueBuffer(buf)

        print("Starting acquisition...")
        data_stream.StartAcquisition()
        try:
            remote_nodemap.FindNode("AcquisitionStart").Execute()
            remote_nodemap.FindNode("AcquisitionStart").WaitUntilDone()
        except Exception:
            pass

        print("Ready. Press SPACE to capture, 'q' to quit.")
        idx = next_image_index(DATA_DIR)

    # capture loop
        running = True
        while running:
            print("Waiting for key... (SPACE to capture, q to quit)")
            k = wait_for_key()
            if k == "q":
                print("Quitting...")
                running = False
                break
            if k == " ":
                try:
                    # Wait for a finished buffer and save it
                    buffer = data_stream.WaitForFinishedBuffer(2000)
                    img = ids_peak_ipl_extension.BufferToImage(buffer)
                    
                    # Convert to RGB8
                    color_image = img.ConvertTo(ids_ipl.PixelFormatName_RGB8)
                    
                    # Get numpy array
                    arr = color_image.get_numpy_3D()
                    
                    # Apply white balance correction (removes green tint)
                    arr = arr.astype(np.float32)
                    arr[:, :, 0] *= 1  # Boost red channel
                    arr[:, :, 1] *= 1  # Boost green channel
                    arr[:, :, 2] *= 4  # Boost blue channel
                    arr = np.clip(arr, 0, 255)
                    
                    # Apply gamma correction (brightens image naturally)
                    arr = arr / 255.0
                    arr = np.power(arr, 1/2.2)
                    arr = (arr * 255).astype(np.uint8)
                    
                    # Convert to BGR and save
                    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                    filename = f"image_{idx:04d}.png"
                    out_path = os.path.join(DATA_DIR, filename)
                    cv2.imwrite(out_path, bgr)
                    
                    print(f"Saved {out_path}")
                    idx += 1
                    
                    # Re-queue the buffer so it can be reused
                    data_stream.QueueBuffer(buffer)
                except Exception as e:
                    print(f"Capture failed: {e}")
                    time.sleep(0.1)

            else:
                # ignore other keys
                continue

    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        try:
            print("Stopping acquisition and cleaning up...")
            try:
                remote_nodemap.FindNode("AcquisitionStop").Execute()
                remote_nodemap.FindNode("AcquisitionStop").WaitUntilDone()
            except Exception:
                pass
            data_stream.StopAcquisition(ids_peak.AcquisitionStopMode_Default)
            data_stream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)
            for b in data_stream.AnnouncedBuffers():
                try:
                    data_stream.RevokeBuffer(b)
                except Exception:
                    pass
            try:
                remote_nodemap.FindNode("TLParamsLocked").SetValue(0)
            except Exception:
                pass
        except Exception:
            pass
        finally:
            ids_peak.Library.Close()


if __name__ == "__main__":
    sys.exit(main() or 0)
