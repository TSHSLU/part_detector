"""
Camera Capture Module
Handles camera initialization, image capture, and camera cleanup.
"""

import numpy as np
import cv2
import os
import time

# Try importing IDS uEye bindings. If unavailable, we'll fallback to OpenCV.
try:
    from ids_peak import ids_peak as ids_peak
    from ids_peak import ids_peak_ipl_extension as ids_peak_ipl_extension
    import ids_peak_ipl.ids_peak_ipl as ids_ipl
    IDS_AVAILABLE = True
except Exception:
    IDS_AVAILABLE = False

# Default camera settings file (optional)
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "camsettings.cset")


class CameraCapture:
    """
    Handles camera operations for capturing images.
    This class provides a structure for camera initialization and image capture.
    """
    
    def __init__(self, camera_id=0):
        """
        Initialize the camera capture.
        
        Args:
            camera_id (int): Camera device ID or connection identifier
        """
        self.camera_id = camera_id
        self.camera = None
        self.is_initialized = False
        # IDS-specific members
        self.use_ids = False
        self.ids_device = None
        self.ids_data_stream = None
        self.ids_buffers = []
        self.remote_nodemap = None
        
    def initialize(self):
        """
        Initialize the camera and configure settings.
        
        Returns:
            bool: True if initialization was successful, False otherwise
            
        TODO: Implement camera initialization logic
        - Connect to the camera
        - Load camera settings (e.g., from camsettings.cset)
        - Configure camera parameters (exposure, gain, etc.)
        - Set up any required buffers or resources
        """
        # Try to initialize IDS camera first (if bindings are available)
        if IDS_AVAILABLE:
            try:
                ids_peak.Library.Initialize()

                device_manager = ids_peak.DeviceManager.Instance()
                device_manager.Update()

                # If there is at least one IDS device, open the first one
                if not device_manager.Devices().empty():
                    device = device_manager.Devices()[0].OpenDevice(ids_peak.DeviceAccessType_Control)
                    self.ids_device = device
                    try:
                        self.remote_nodemap = device.RemoteDevice().NodeMaps()[0]
                    except Exception:
                        self.remote_nodemap = None

                    # Try loading saved settings if the file exists
                    if self.remote_nodemap is not None and os.path.exists(CONFIG_FILE):
                        try:
                            self.remote_nodemap.LoadFromFile(CONFIG_FILE)
                            print("INFO: Loaded camera settings from", CONFIG_FILE)
                        except Exception:
                            # ignore failures and continue with defaults
                            pass

                    # Open data stream and allocate buffers
                    data_stream = device.DataStreams()[0].OpenDataStream()
                    self.ids_data_stream = data_stream

                    # Payload size
                    try:
                        payload_size = self.remote_nodemap.FindNode("PayloadSize").Value()
                    except Exception:
                        payload_size = 0

                    # Allocate minimum required buffers
                    try:
                        buffer_count = data_stream.NumBuffersAnnouncedMinRequired()
                    except Exception:
                        buffer_count = 4

                    # Announce and queue buffers
                    for _ in range(buffer_count):
                        try:
                            buf = data_stream.AllocAndAnnounceBuffer(payload_size)
                            self.ids_buffers.append(buf)
                        except Exception:
                            # If allocation fails, continue with what we have
                            print("WARNING: Buffer allocation failed")
                            break

                    for b in list(self.ids_buffers):
                        try:
                            data_stream.QueueBuffer(b)
                        except Exception:
                            # ignore individual queue failures
                            pass

                    # Start acquisition
                    try:
                        data_stream.StartAcquisition()
                        try:
                            if self.remote_nodemap is not None:
                                self.remote_nodemap.FindNode("AcquisitionStart").Execute()
                                self.remote_nodemap.FindNode("AcquisitionStart").WaitUntilDone()
                                print("INFO: Camera acquisition started")
                        except Exception:
                            pass
                    except Exception:
                        # If acquisition fails, fall back to OpenCV
                        raise

                    self.use_ids = True
                    self.is_initialized = True
                    return True

            except Exception:
                # Cleanup partial IDS init and fall back to OpenCV below
                try:
                    if self.ids_data_stream is not None:
                        try:
                            self.ids_data_stream.StopAcquisition(ids_peak.AcquisitionStopMode_Default)
                        except Exception:
                            pass
                except Exception:
                    pass

        # Fallback: use OpenCV VideoCapture
        try:
            # On Windows try DirectShow backend for better compatibility
            self.camera = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        except Exception:
            self.camera = cv2.VideoCapture(self.camera_id)

        # Try to set a sensible default resolution (may be ignored)
        try:
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        except Exception:
            pass

        if self.camera is None or not self.camera.isOpened():
            print("ERROR: Could not initialize any camera (no IDS device and no system camera available)")
            self.is_initialized = False
            return False

        self.is_initialized = True
        self.use_ids = False
        return True
    
    def capture_frame(self):
        """
        Capture a single frame from the camera.
        
        Returns:
            numpy.ndarray: Captured image as BGR numpy array, or None if capture failed
            
        TODO: Implement frame capture logic
        - Grab frame from camera
        - Convert to appropriate format (BGR)
        - Handle any camera-specific processing
        - Return the image as numpy array
        """
        if not self.is_initialized:
            return None

        # IDS capture path
        if self.use_ids and self.ids_data_stream is not None:
            try:
                # Wait for finished buffer
                buffer = self.ids_data_stream.WaitForFinishedBuffer(2000)
                img = ids_peak_ipl_extension.BufferToImage(buffer)

                # Convert to RGB8 and get numpy array
                color_image = img.ConvertTo(ids_ipl.PixelFormatName_RGB8)
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
                    

                # Convert RGB -> BGR for OpenCV
                bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

                # Re-queue buffer for reuse
                try:
                    self.ids_data_stream.QueueBuffer(buffer)
                except Exception:
                    pass

                return bgr
            except Exception:
                # On any failure return None
                return None

        # OpenCV fallback
        if self.camera is not None:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    return None
                return frame
            except Exception:
                return None

        return None
    
    def get_camera_info(self):
        """
        Get camera information and current settings.
        
        Returns:
            dict: Dictionary containing camera information
            
        TODO: Implement camera info retrieval
        - Get camera model/name
        - Get current resolution
        - Get current exposure, gain, etc.
        - Return as dictionary
        """
        info = {
            "model": "Unknown",
            "resolution": None,
            "status": "not initialized"
        }

        if not self.is_initialized:
            return info

        info["status"] = "initialized"

        if self.use_ids and self.ids_device is not None:
            try:
                # Try to read a model name from the nodemap if present
                if self.remote_nodemap is not None:
                    try:
                        name_node = self.remote_nodemap.FindNode("DeviceModelName")
                        info["model"] = str(name_node.Value())
                    except Exception:
                        info["model"] = "IDS uEye"

                # Try to estimate resolution from payload or camera properties
                try:
                    w_node = self.remote_nodemap.FindNode("Width")
                    h_node = self.remote_nodemap.FindNode("Height")
                    info["resolution"] = (int(w_node.Value()), int(h_node.Value()))
                except Exception:
                    info["resolution"] = None
            except Exception:
                pass
        elif self.camera is not None:
            try:
                w = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                info["resolution"] = (w, h)
                info["model"] = f"OpenCV Camera {self.camera_id}"
            except Exception:
                pass

        return info
    
    def release(self):
        """
        Release camera resources and cleanup.
        
        TODO: Implement cleanup logic
        - Stop camera capture
        - Release camera resources
        - Clean up any buffers or handles
        """
        # IDS cleanup
        if self.use_ids and self.ids_data_stream is not None:
            try:
                try:
                    if self.remote_nodemap is not None:
                        try:
                            self.remote_nodemap.FindNode("AcquisitionStop").Execute()
                            self.remote_nodemap.FindNode("AcquisitionStop").WaitUntilDone()
                        except Exception:
                            pass

                    self.ids_data_stream.StopAcquisition(ids_peak.AcquisitionStopMode_Default)
                    self.ids_data_stream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)
                except Exception:
                    pass

                # Revoke buffers
                for b in list(self.ids_data_stream.AnnouncedBuffers()):
                    try:
                        self.ids_data_stream.RevokeBuffer(b)
                    except Exception:
                        pass
            except Exception:
                pass

            # Close IDS library
            try:
                ids_peak.Library.Close()
            except Exception:
                pass

            self.ids_data_stream = None
            self.ids_buffers = []
            self.ids_device = None
            self.use_ids = False

        # OpenCV cleanup
        if self.camera is not None:
            try:
                self.camera.release()
            except Exception:
                pass
            self.camera = None

        self.is_initialized = False
        
    def __enter__(self):
        """Context manager entry - initializes camera."""
        self.initialize()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - releases camera."""
        self.release()
