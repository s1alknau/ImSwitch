import numpy as np
import time
from imswitch.imcommon.model import initLogger
from skimage.filters import gaussian, median
from typing import List
import sys
import atexit
import weakref
from ctypes import *
import collections

from sys import platform
try:
    if platform == "linux" or platform == "linux2":
        # linux
        from imswitch.imcontrol.model.interfaces.hikrobotMac.MvCameraControl_class import *
    elif platform == "darwin":
        # OS X
        from imswitch.imcontrol.model.interfaces.hikrobotMac.MvCameraControl_class import *
        pass
    elif platform == "win32":
        from imswitch.imcontrol.model.interfaces.hikrobotWin.MvCameraControl_class import *
except Exception as e:
    print(e)

# Pixel format constants — 8-bit
PixelType_Gvsp_Mono8 = 17301505
PixelType_Gvsp_Mono8_Signed = 17301506
PixelType_Gvsp_RGB8_Packed = 35127316
PixelType_Gvsp_BayerRG8 = 17301512
PixelType_Gvsp_BayerBG8 = 17301513
PixelType_Gvsp_BayerGB8 = 17301514
PixelType_Gvsp_BayerGR8 = 17301515

# Pixel format constants — 10/12/16-bit (from SDK headers)
PixelType_Gvsp_Mono10 = 17825795
PixelType_Gvsp_Mono12 = 17825797
PixelType_Gvsp_Mono16 = 17825799
PixelType_Gvsp_Mono10_Packed = 17563652
PixelType_Gvsp_Mono12_Packed = 17563654
PixelType_Gvsp_BayerRG10 = 17825805
PixelType_Gvsp_BayerRG12 = 17825809

# Set of pixel types that deliver >8-bit mono data (2 bytes per pixel, unpacked)
_MONO_HIGHBIT_FORMATS = {
    PixelType_Gvsp_Mono10,
    PixelType_Gvsp_Mono12,
    PixelType_Gvsp_Mono16,
}

# Bit-packed mono formats that need SDK conversion before use.
# Mono10_Packed = 4 pixels in 5 bytes, Mono12_Packed = 2 pixels in 3 bytes
# — these CANNOT be interpreted as plain uint16 arrays.
_MONO_PACKED_FORMATS = {
    PixelType_Gvsp_Mono10_Packed,
    PixelType_Gvsp_Mono12_Packed,
}

# Set of pixel types that deliver >8-bit Bayer data
_BAYER_HIGHBIT_FORMATS = {
    PixelType_Gvsp_BayerRG10,
    PixelType_Gvsp_BayerRG12,
}

# Some possible YUV pixel formats:
PixelType_Gvsp_YUV444_Packed = 35127328
PixelType_Gvsp_YUV422_YUYV_Packed = 34603058
PixelType_Gvsp_YUV422_Packed = 34603039
PixelType_Gvsp_YUV411_Packed = 34340894


# ----------------------------------------------------------------------------
#  Callback signature
# ----------------------------------------------------------------------------
CALLBACK_SIG = CFUNCTYPE(
    None,
    POINTER(c_ubyte),               # pUserBuf
    POINTER(MV_FRAME_OUT_INFO_EX),  # pFrameInfo
    c_void_p                        # pUser (void*)
)
# ----------------------------------------------------------------------------
def _close_camera_on_exit(camera_ref):
    """
    Release a camera that is still open when the interpreter shuts down.

    A GigE camera keeps its control channel assigned to the process that
    opened it. If that process ends without MV_CC_CloseDevice, every later
    OpenDevice fails with MV_E_ACCESS_DENIED (0x80000203) and the camera has
    to be power-cycled before it can be used again - it looks exactly like a
    camera that is no longer detected. Registered per instance via atexit,
    with a weak reference so the hook never keeps the camera alive by itself.
    """
    camera = camera_ref()
    if camera is None:
        return
    try:
        camera.close()
    except Exception:
        pass  # interpreter is going down; nothing useful left to report


class CameraHIK:
    """Minimal wrapper that grabs frames via SDK callback (no polling)."""

    def __init__(self,cameraNo=None, exposure_time = 10000, gain = 0, frame_rate=30, blacklevel=100, isRGB=False, binning=2, flipImage=(False, False)):
        super().__init__()
        self.__logger = initLogger(self, tryInheritParent=False)

        self.model = "CameraHIK"
        self.shape = (0, 0)
        self.is_connected = False
        self.is_streaming = False
        self.downsamplepreview = 1

        self.blacklevel = blacklevel
        self.exposure_time = exposure_time
        self.gain = gain
        self.frame_rate = frame_rate
        self.cameraNo = cameraNo
        self.flipImage = flipImage  # (flipY, flipX)

        self.NBuffer = 3
        self.frame_buffer = collections.deque(maxlen=self.NBuffer)
        self.frameid_buffer = collections.deque(maxlen=self.NBuffer)
        self.flatfieldImage = None
        self.camera = None
        self.DEBUG = False
        # Binning
        if platform in ("darwin", "linux2", "linux"):
            binning = 2
        self.binning = binning

        self.SensorHeight = 0
        self.SensorWidth = 0
        self.frame = np.zeros((self.SensorHeight, self.SensorWidth))

        self.lastFrameFromBuffer = None
        self.lastFrameId = -1
        self.frameNumber = -1
        self.g_bExit = False

        self._open_camera(self.cameraNo)
        atexit.register(_close_camera_on_exit, weakref.ref(self))

        self.isFlatfielding = False

        # use the parameter passed to __init__, or fall back to auto-detect
        if isRGB is not None:
            self.isRGB = bool(isRGB)
        else:
            # fall back to auto-detect from camera parameters
            detected_rgb = self.mParameters.get("isRGB", False)
            self.isRGB = bool(detected_rgb) if detected_rgb is not None else self._detect_rgb()

        self.__logger.info(f"Camera RGB mode: {self.isRGB}")

        # Set pixel format based on RGB mode
        self._activePixelFormat = PixelType_Gvsp_Mono8  # default
        if self.isRGB:
            # Try different RGB/YUV formats for color cameras
            formats_to_try = [
                PixelType_Gvsp_RGB8_Packed,
                PixelType_Gvsp_YUV422_YUYV_Packed,
                PixelType_Gvsp_BayerRG8,
                PixelType_Gvsp_BayerBG8,
                PixelType_Gvsp_BayerGB8,
                PixelType_Gvsp_BayerGR8
            ]

            format_set = False
            for pixel_format in formats_to_try:
                ret = self.camera.MV_CC_SetEnumValue("PixelFormat", pixel_format)
                if ret == 0:
                    self.__logger.info(f"Successfully set pixel format to 0x{pixel_format:x} for RGB camera")
                    self._activePixelFormat = pixel_format
                    format_set = True
                    break
                else:
                    self.__logger.info(f"Failed to set pixel format 0x{pixel_format:x}, ret=0x{ret:x}")

            if not format_set:
                self.__logger.warning("Could not set any RGB pixel format, camera may not work correctly")
        else:
            # For mono cameras, try highest bit-depth first, fall back to Mono8
            mono_formats_to_try = [
                (PixelType_Gvsp_Mono12, "Mono12"),
                (PixelType_Gvsp_Mono10, "Mono10"),
                (PixelType_Gvsp_Mono8, "Mono8"),
            ]
            format_set = False
            for pixel_format, name in mono_formats_to_try:
                ret = self.camera.MV_CC_SetEnumValue("PixelFormat", pixel_format)
                if ret == 0:
                    self.__logger.info(f"Set pixel format to {name} for monochrome camera")
                    self._activePixelFormat = pixel_format
                    format_set = True
                    break
                else:
                    self.__logger.debug(f"Mono format {name} not supported, ret=0x{ret:x}")
            if not format_set:
                self.__logger.warning("Could not set any mono pixel format")
                self._activePixelFormat = PixelType_Gvsp_Mono8

        # Mark camera as connected after successful initialization
        self.is_connected = True

        # register callback -----------------------------------------------
        self._sdk_cb = self._wrap_cb(self._on_frame)   # keep ref
        ret = self.camera.MV_CC_RegisterImageCallBackEx(self._sdk_cb, None)
        if ret != 0:
            self.is_connected = False
            raise RuntimeError(f"Register cb failed 0x{ret:x}")

        # Track callback registration state
        self._callback_registered = True


    # ---------------------------------------------------------------------
    # Camera discovery / opening
    # ---------------------------------------------------------------------
    def _open_camera(self, number: int):
        # gather all devices (GigE then USB)
        infos = []  # List[MV_CC_DEVICE_INFO]
        for layer in (MV_GIGE_DEVICE, MV_USB_DEVICE):
            lst = MV_CC_DEVICE_INFO_LIST()
            # print available cameras
            self.__logger.debug(f"Searching for cameras on layer {layer}...")
            memset(byref(lst), 0, sizeof(lst))
            if MvCamera.MV_CC_EnumDevices(layer, lst) == 0:
                for i in range(lst.nDeviceNum):
                    device_info = cast(lst.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
                    infos.append(device_info)
                    self.__logger.debug(f"Found camera {i}: Layer {layer}")

        self.__logger.info(f"Total cameras found: {len(infos)}")
        if not infos:
            raise RuntimeError("No suitable Hik cameras found.")

        # If camera number > 9, treat it as a unique identifier (serial number checksum)
        if number > 9:
            self.__logger.info(f"Camera number {number} > 9, treating as unique identifier (serial checksum)")
            camera_index = None
            serial_checksums = []
            # Search for camera with matching serial number checksum
            for i, info in enumerate(infos):
                serial_checksum = np.sum(info.SpecialInfo.stUsb3VInfo.chSerialNumber)
                serial_checksums.append(serial_checksum)
                self.__logger.info(f"Camera {i} serial checksum: {serial_checksum}")
                if serial_checksum == number:
                    camera_index = i
                    self.__logger.info(f"Found camera with matching serial checksum {number} at index {i}")
                    break

            if camera_index is None:
                # Fallback: use the first available camera if requested checksum not found
                # This handles the case where camera order may have changed or camera was replaced
                self.__logger.warning(
                    f"No camera found with serial checksum {number}. "
                    f"Falling back to first available camera (index 0)."
                )
                # Log available cameras for diagnostic purposes
                # TODO: Fallback -> We need to open another camera, but there may be a conflict with another instance needing that camera, currently we don't have any way to catch this/read this from the config
                # FIXME: For now: iterate through list and check if opened, if not, use that index
                for i, info in enumerate(infos):
                    tmpCamera = MvCamera()
                    tmpCamera.MV_CC_CreateHandle(info)
                    ret = tmpCamera.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
                    if ret == 0:
                        serial_checksum = np.sum(info.SpecialInfo.stUsb3VInfo.chSerialNumber)
                        self.__logger.info(f"Camera {i} serial checksum is available and will be used: {serial_checksum}")
                        camera_index = i                
                        tmpCamera.MV_CC_CloseDevice()
                        tmpCamera.MV_CC_DestroyHandle()
                        del tmpCamera
                        break
            number = camera_index  # Use the found index for the rest of the function
        else:
            # Traditional index-based selection
            if number >= len(infos):
                self.__logger.warning(
                    f"Requested camera index {number} out of range (only {len(infos)} cameras available). "
                    f"Falling back to camera index 0."
                )
                number = 0

        self.__logger.info(f"Opening camera {number} out of {len(infos)} available cameras")

        # Track the serial checksum of the camera we're opening
        self._serial_checksum = np.sum(infos[number].SpecialInfo.stUsb3VInfo.chSerialNumber)

        self.camera = MvCamera()
        ret = self.camera.MV_CC_CreateHandle(infos[number])
        if ret != 0:
            raise RuntimeError(f"CreateHandle failed 0x{ret:x}")
        ret = self.camera.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            raise RuntimeError(f"OpenDevice failed 0x{ret:x}")

        # optimise packet size for GigE
        if infos[number].nTLayerType == MV_GIGE_DEVICE:
            psize = self.camera.MV_CC_GetOptimalPacketSize()
            if psize > 0:
                self.camera.MV_CC_SetIntValue("GevSCPSPacketSize", psize)
                self.__logger.debug(f"Set packet size to {psize} for GigE camera")
        # print unique ID: # TODO: We should make the cameraNo persistent based on this ID
        self.__logger.info(f"Unique Serial Number of HIK Camera: {np.sum(infos[number].SpecialInfo.stUsb3VInfo.chSerialNumber)}")
        # get available parameters
        self.mParameters = self.get_camera_parameters()
        self.__logger.info(f"Camera parameters: model={self.mParameters.get('model_name', 'Unknown')}, isRGB={self.mParameters.get('isRGB', False)}")

        # set parameters
        self.setBinning(binning=self.binning)
        self.trigger_source = self.mParameters.get("trigger_source", "Continuous")
        # TODO: This is necessary to ensure that we don't create delays when operating multiple cameras
        stBool = c_bool(False)
        ret = self.camera.MV_CC_GetBoolValue("AcquisitionFrameRateEnable", stBool)
        if ret != 0:
            self.__logger.debug("Get AcquisitionFrameRateEnable fail! ret[0x%x]" % ret)
        # TODO: Limit our framerate (force) to 30 fps for now
        if self.frame_rate > 0:
            stBool.value = True
            ret = self.camera.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", stBool)
            if ret != 0:
                self.__logger.debug("Set AcquisitionFrameRateEnable fail! ret[0x%x]" % ret)
            ret = self.camera.MV_CC_SetFloatValue("AcquisitionFrameRate", self.frame_rate)
            if ret != 0:
                self.__logger.debug("Set AcquisitionFrameRate fail! ret[0x%x]" % ret)
            self.__logger.info(f"Set frame rate to {self.frame_rate} fps")
        else:
            stBool.value = False
            ret = self.camera.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", stBool)
            if ret != 0:
                self.__logger.debug("Set AcquisitionFrameRateEnable fail! ret[0x%x]" % ret)
        ret = self.camera.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            self.__logger.debug("Set trigger mode fail! ret[0x%x]" % ret)
            sys.exit()

        # setup sensor size
        mWidth = MVCC_INTVALUE()
        mHeight = MVCC_INTVALUE()
        self.camera.MV_CC_GetIntValue("WidthMax", mWidth)
        self.camera.MV_CC_GetIntValue("HeightMax", mHeight)
        self.SensorWidth = mWidth.nCurValue
        self.SensorHeight = mHeight.nCurValue

        print(f"Current number of pixels: Width = {self.SensorWidth}, Height = {self.SensorHeight}")

    def reconnectCamera(self):
        # Safely close any existing handle
        # todo: Need to store the current camerano and other settings and open it in case it's the hash/referenc enumber
        if self.camera is not None:
            try:
                # Deregister callback if registered
                if hasattr(self, '_callback_registered') and self._callback_registered:
                    self.camera.MV_CC_RegisterImageCallBackEx(None, None)
                    self._callback_registered = False

                self.camera.MV_CC_CloseDevice()
                self.camera.MV_CC_DestroyHandle()
            except Exception as e:
                self.__logger.error(f"Error while closing camera handle: {e}")
            self.camera = None

        # Re-initialize camera with original cameraNo
        try:
            self._open_camera(number=self.cameraNo)
            self.__logger.debug("Camera reconnected successfully.")
        except Exception as e:
            self.__logger.error(f"Failed to reconnect camera: {e}")

    # ---------------------------------------------------------------------
    # RGB detection (very rough – checks model string for "UC")
    # ---------------------------------------------------------------------
    def _detect_rgb(self) -> bool:
        name = MVCC_STRINGVALUE(); self.camera.MV_CC_GetStringValue("DeviceModelName", name)
        return "UC" in name.chCurValue.decode()

    def getTriggerTypes(self) -> List[str]:
        """Return a list of available trigger types."""
        if not self.is_connected:
            return ["Camera not connected"]
        return [
            "Continuous (Free Run)",
            "Software Trigger",
            "External Trigger (LINE0)"
        ]

    def getTriggerSource(self) -> str:
        """Return the current trigger source as a string."""
        if not self.is_connected:
            return "Camera not connected"
        if self.trigger_source == MV_TRIGGER_SOURCE_SOFTWARE:
            return "Software Trigger"
        elif self.trigger_source == MV_TRIGGER_SOURCE_LINE0:
            return "External Trigger (LINE0)"
        else:
            return "Continuous (Free Run)"
    # ---------------------------------------------------------------------
    # C callback factory ---------------------------------------------------
    # ---------------------------------------------------------------------
    def _on_frame(self, frame: np.ndarray, fid: int, ts: int):
        # The array handed in is a zero-copy view onto the SDK's frame buffer,
        # which the camera reuses for the frames that follow. Storing the view
        # means every consumer holds a window that keeps changing underneath
        # it: a frame read as bright is dark again milliseconds later, and a
        # read that overlaps an incoming frame returns one image stitched
        # together from two. The ring buffer therefore keeps its own copy.
        self.frame_buffer.append(np.array(frame, copy=True))
        self.frameid_buffer.append(fid)
        self.frameNumber = fid
        self.timestamp   = ts
        if self.DEBUG:
            print("frame received:", fid, "timestamp:", ts, "mean:", np.mean(frame), "from camera name: ", self.mParameters.get("model_name", "Unknown"))

    def _wrap_cb(self, user_cb):
        '''
        Wrap a user callback function to be used with the SDK.
        The main task is to convert the SDK buffer into a NumPy array and
        pass it to the user callback.'''
        @CALLBACK_SIG
        def _cb(pData, pInfo, _):
            info = pInfo.contents
            w, h   = info.nWidth, info.nHeight
            nSize  = info.nFrameLen
            pix    = info.enPixelType
            fid    = info.nFrameNum
            ts     = self._hw_timestamp(info)        # ← fixed

            # build NumPy view over the SDK buffer (zero-copy)
            # For >8-bit unpacked mono formats, each pixel is 2 bytes (uint16)
            if pix in _MONO_HIGHBIT_FORMATS:
                buf = np.frombuffer(
                    (c_ubyte * nSize).from_address(addressof(pData.contents)),
                    dtype=np.uint16
                )
            elif pix in _MONO_PACKED_FORMATS:
                # Bit-packed formats need SDK conversion to Mono16 first
                nDst = w * h * 2  # 2 bytes per pixel in Mono16
                dst  = (c_ubyte * nDst)()
                conv = MV_CC_PIXEL_CONVERT_PARAM()
                memset(byref(conv), 0, sizeof(conv))
                conv.nWidth         = w
                conv.nHeight        = h
                conv.enSrcPixelType = pix
                conv.enDstPixelType = PixelType_Gvsp_Mono16
                conv.pSrcData       = pData
                conv.nSrcDataLen    = nSize
                conv.pDstBuffer     = dst
                conv.nDstBufferSize = nDst
                ret = self.camera.MV_CC_ConvertPixelType(conv)
                if ret != 0:
                    self.__logger.error(f"Packed mono convert failed 0x{ret:x}")
                    return
                buf = np.frombuffer(dst, dtype=np.uint16, count=w * h)
            else:
                buf = np.frombuffer(
                    (c_ubyte * nSize).from_address(addressof(pData.contents)),
                    dtype=np.uint8
                )

            # reshape according to pixel type
            if pix in _MONO_HIGHBIT_FORMATS or pix in _MONO_PACKED_FORMATS:  # mono 10/12/16-bit
                frame = buf.reshape(h, w)
            elif pix == PixelType_Gvsp_Mono8:              # mono 8-bit
                frame = buf.reshape(h, w)
            elif pix in (PixelType_Gvsp_RGB8_Packed,
                        PixelType_Gvsp_BayerRG8,
                        PixelType_Gvsp_BayerBG8,
                        PixelType_Gvsp_BayerGB8,
                        PixelType_Gvsp_BayerGR8):
                # convert to RGB for non-packed types
                if pix != PixelType_Gvsp_RGB8_Packed:
                    nRGB = w * h * 3
                    dst  = (c_ubyte * nRGB)()
                    conv = MV_CC_PIXEL_CONVERT_PARAM()
                    memset(byref(conv), 0, sizeof(conv))
                    conv.nWidth         = w
                    conv.nHeight        = h
                    conv.enSrcPixelType = pix
                    conv.enDstPixelType = PixelType_Gvsp_RGB8_Packed
                    conv.pSrcData       = pData
                    conv.nSrcDataLen    = nSize
                    conv.pDstBuffer     = dst
                    conv.nDstBufferSize = nRGB
                    ret = self.camera.MV_CC_ConvertPixelType(conv)
                    if ret != 0:
                        self.__logger.error(f"Pixel convert failed 0x{ret:x}")
                        return
                    buf   = np.frombuffer(dst, dtype=np.uint8, count=nRGB)
                frame = buf.reshape(h, w, 3)
            elif pix in (PixelType_Gvsp_YUV422_YUYV_Packed,
                        PixelType_Gvsp_YUV422_Packed,
                        PixelType_Gvsp_YUV444_Packed,
                        PixelType_Gvsp_YUV411_Packed):
                # convert YUV to RGB
                nRGB = w * h * 3
                dst  = (c_ubyte * nRGB)()
                conv = MV_CC_PIXEL_CONVERT_PARAM()
                memset(byref(conv), 0, sizeof(conv))
                conv.nWidth         = w
                conv.nHeight        = h
                conv.enSrcPixelType = pix
                conv.enDstPixelType = PixelType_Gvsp_RGB8_Packed
                conv.pSrcData       = pData
                conv.nSrcDataLen    = nSize
                conv.pDstBuffer     = dst
                conv.nDstBufferSize = nRGB
                ret = self.camera.MV_CC_ConvertPixelType(conv)
                if ret != 0:
                    self.__logger.error(f"YUV pixel convert failed 0x{ret:x}")
                    return
                buf = np.frombuffer(dst, dtype=np.uint8, count=nRGB)
                frame = buf.reshape(h, w, 3)
                self.__logger.debug(f"Converted YUV format 0x{pix:x} to RGB")
            else:
                self.__logger.error(f"Unsupported pixel type 0x{pix:x}")
                return

            # Apply flip if needed (zero-CPU operation using numpy)
            if self.flipImage[0]:  # flipY
                frame = np.flip(frame, axis=0)
            if self.flipImage[1]:  # flipX
                frame = np.flip(frame, axis=1)


            # pass to user callback
            user_cb(frame, fid, ts)

        return _cb

    def get_camera_parameters(self):
        param_dict = {}

        # PixelFormat and check if color
        stPixelFormat = MVCC_ENUMVALUE()
        ret = self.camera.MV_CC_GetEnumValue("PixelFormat", stPixelFormat)
        if ret == 0:
            param_dict["pixel_format"] = stPixelFormat.nCurValue

        # camera Name
        stName = MVCC_STRINGVALUE()
        param_dict["isRGB"] = False
        ret = self.camera.MV_CC_GetStringValue("DeviceModelName", stName)
        if ret == 0:
            param_dict["model_name"] = stName.chCurValue.decode("utf-8")
            if param_dict["model_name"].find("UC")>0:
                param_dict["isRGB"] = True
        # if isRGB switch off AWB
        if param_dict["isRGB"] and False: # TODO: Review
            ret = self.camera.MV_CC_SetEnumValue("BalanceWhiteAuto", MV_BALANCEWHITE_AUTO_CONTINUOUS)
            if ret != 0:
                print("set BalanceWhiteAuto failed! ret [0x%x]" % ret)
                self.init_ok = False
            else:
                # 2) Set color temperature mode to Wide
                ret = self.camera.MV_CC_SetEnumValueByString("BalanceColorTempMode", "WideMode")
                if ret != 0:
                    print("set BalanceColorTempMode failed! ret [0x%x]" % ret)
        elif param_dict["isRGB"]:
            ret = self.camera.MV_CC_SetEnumValue("BalanceWhiteAuto", MV_BALANCEWHITE_AUTO_OFF)
            if ret != 0:
                print("set BalanceWhiteAuto failed! ret [0x%x]" % ret)


        # Image Width
        stWidth = MVCC_INTVALUE()
        ret = self.camera.MV_CC_GetIntValue("Width", stWidth)
        if ret == 0:
            param_dict["width"] = stWidth.nCurValue

        # Image Height
        stHeight = MVCC_INTVALUE()
        ret = self.camera.MV_CC_GetIntValue("Height", stHeight)
        if ret == 0:
            param_dict["height"] = stHeight.nCurValue

        # get exposure time
        mExposureValues = self.get_exposuretime()
        if mExposureValues[0] is not None:
            param_dict["exposure_current"] = mExposureValues[0]
            param_dict["exposure_min"] = mExposureValues[1]
            param_dict["exposure_max"] = mExposureValues[2]

        # get gain
        mGainValues = self.get_gain()
        if mGainValues[0] is not None:
            param_dict["gain_current"] = mGainValues[0]
            param_dict["gain_min"] = mGainValues[1]
            param_dict["gain_max"] = mGainValues[2]

        return param_dict

    def get_gain(self):
        # Current / Min / Max Gain
        stGain = MVCC_FLOATVALUE()
        ret = self.camera.MV_CC_GetFloatValue("Gain", stGain)
        if ret == 0:
            return (stGain.fCurValue, stGain.fMin, stGain.fMax)
        else:
            self.__logger.error(f"Get gain failed 0x{ret:x}")
            return (None, None, None)


    def get_exposuretime(self):
        # Current / Min / Max Exposure
        stExposure = MVCC_FLOATVALUE()
        ret = self.camera.MV_CC_GetFloatValue("ExposureTime", stExposure)
        if ret == 0:
            return (stExposure.fCurValue, stExposure.fMin, stExposure.fMax)
        else:
            self.__logger.error(f"Get exposure time failed 0x{ret:x}")
            return (None, None, None)


    def start_live(self):
        if self.is_streaming:
            return
        self.flushBuffer()

        # in case the camera is None, we have to restart it anyway
        if self.camera is None: # TODO: 2025-11-03 15:13:38 ERROR [LiveViewController] Error starting live view: 'NoneType' object has no attribute 'MV_CC_RegisterImageCallBackEx'
            self.reconnectCamera()
            ret = self.camera.MV_CC_StartGrabbing()
            if ret != 0:
                raise RuntimeError(f"StartGrabbing failed 0x{ret:x}")
            self.is_streaming = True
            return

        # Re-register callback if needed (in case it was deregistered during stop)
        if hasattr(self, '_callback_registered') and not self._callback_registered:
            ret = self.camera.MV_CC_RegisterImageCallBackEx(self._sdk_cb, None)
            if ret != 0:
                raise RuntimeError(f"Re-register callback failed 0x{ret:x}")
            self._callback_registered = True
            self.__logger.debug("Callback re-registered successfully")

        try:
            ret = self.camera.MV_CC_StartGrabbing()
            if ret != 0:
                #raise RuntimeError(f"StartGrabbing failed 0x{ret:x}")
                self.reconnectCamera()
                ret = self.camera.MV_CC_StartGrabbing()
                if ret != 0:
                    self.__logger.error(f"StartGrabbing exception: RuntimeError after reconnect 0x{ret:x}")
                    raise RuntimeError(f"StartGrabbing failed after reconnect 0x{ret:x}")
        except Exception:
            raise RuntimeError(f"StartGrabbing failed 0x{ret:x}")
        self.is_streaming = True

    def stop_live(self):
        if not self.is_streaming:
            return

        # Stop grabbing first
        self.camera.MV_CC_StopGrabbing()

        # Deregister callback to ensure clean state for next start
        if hasattr(self, '_callback_registered') and self._callback_registered:
            ret = self.camera.MV_CC_RegisterImageCallBackEx(None, None)
            if ret == 0:
                self._callback_registered = False
                self.__logger.debug("Callback deregistered successfully")
            else:
                self.__logger.warning(f"Failed to deregister callback: 0x{ret:x}")

        self.is_streaming = False

    def suspend_live(self):
        self.stop_live()

    def prepare_live(self):
        pass

    def close(self):
        """Release the camera. Safe to call more than once."""
        if getattr(self, "camera", None) is None:
            return

        try:
            if self.is_streaming:
                self.stop_live()

            # Ensure callback is deregistered before closing
            if hasattr(self, '_callback_registered') and self._callback_registered:
                self.camera.MV_CC_RegisterImageCallBackEx(None, None)
                self._callback_registered = False

            self.camera.MV_CC_CloseDevice()
            self.camera.MV_CC_DestroyHandle()
        finally:
            # Drop the handle even if closing failed, so a second attempt
            # cannot operate on a half-torn-down camera.
            self.camera = None
            self.is_connected = False

    def __del__(self):
        # Backstop for the case where the object is dropped long before
        # interpreter shutdown; atexit alone would hold the camera until then.
        try:
            self.close()
        except Exception:
            pass

    def set_exposure_time(self, exposure_time):
        self.exposure_time = exposure_time
        self.camera.MV_CC_SetFloatValue("ExposureTime", self.exposure_time * 1000)

    def set_exposure_mode(self, exposure_mode="manual"):
        exposure_mode = exposure_mode.lower()
        if exposure_mode == "manual":
            self.camera.MV_CC_SetEnumValue("ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF)
            # Also disable AutoGain — otherwise the SDK default (Continuous)
            # keeps regulating gain and the image "breathes" slowly even
            # though exposure is fixed. The Hik SDK constant MV_GAIN_MODE_OFF
            # is 0, matching MV_EXPOSURE_AUTO_MODE_OFF; reuse that here to
            # avoid an extra import.
            self.camera.MV_CC_SetEnumValue("GainAuto", MV_EXPOSURE_AUTO_MODE_OFF)
        elif exposure_mode == "auto":
            self.camera.MV_CC_SetEnumValue("ExposureAuto", MV_EXPOSURE_AUTO_MODE_CONTINUOUS)
        elif exposure_mode == "once":
            self.camera.MV_CC_SetEnumValue("ExposureAuto", MV_EXPOSURE_AUTO_MODE_ONCE)
        else:
            self.__logger.warning("Exposure mode not recognized")

    def set_camera_mode(self, isAutomatic):
        self.set_exposure_mode(isAutomatic)

    def set_gain(self, gain):
        self.gain = gain
        self.camera.MV_CC_SetFloatValue("Gain", self.gain)

    def set_frame_rate(self, frame_rate):
        ret = self.camera.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
        if ret != 0:
            self.__logger.error("set AcquisitionFrameRateEnable fail! ret[0x%x]" % ret)
        ret = self.camera.MV_CC_SetFloatValue("AcquisitionFrameRate", frame_rate)
        if ret != 0:
            self.__logger.error("set AcquisitionFrameRate fail! ret[0x%x]" % ret)

    def set_flatfielding(self, is_flatfielding):
        self.isFlatfielding = is_flatfielding
        if self.isFlatfielding:
            self.recordFlatfieldImage()

    def setFlatfieldImage(self, flatfieldImage, isFlatfieldEnabeled=True):
        self.flatfieldImage = flatfieldImage
        self.isFlatfielding = isFlatfieldEnabeled

    def set_blacklevel(self, blacklevel):
        self.blacklevel = blacklevel
        self.camera.MV_CC_SetFloatValue("BlackLevel", self.blacklevel)

    def set_pixel_format(self, format):
        # Apply the requested pixel format (or fall back to current)
        if format is None:
            format = getattr(self, '_activePixelFormat', PixelType_Gvsp_Mono8)
        ret = self.camera.MV_CC_SetEnumValue("PixelFormat", format)
        if ret == 0:
            self._activePixelFormat = format
            self.__logger.info(f"Pixel format set to 0x{format:x}")
        else:
            self.__logger.warning(f"Failed to set pixel format 0x{format:x}, ret=0x{ret:x}")

    def setBinning(self, binning=1):
        try:
            self.camera.MV_CC_SetIntValue("BinningX", binning)
            self.camera.MV_CC_SetIntValue("BinningY", binning)
            self.binning = binning
        except Exception as e:
            self.__logger.error(e)

    def getLast(self,
                returnFrameNumber: bool = False,
                timeout: float = 1.0,
                auto_trigger: bool = False): # TODO: This is weird; shall we use it or not?
        """
        Return the newest frame in the ring-buffer.
        If the buffer is empty *and* the camera is in **software-trigger**
        mode, a trigger is fired automatically (once) so the caller does not
        have to worry about it.

        Parameters
        ----------
        returnFrameNumber : bool
            If True return a tuple ``(frame, fid)``.
        timeout : float
            Seconds to wait for a frame before giving up.
        auto_trigger : bool
            Disable if you need manual control over the trigger pulse.
        """
        # one-shot trigger if necessary ---------------------------------------
        if auto_trigger and getattr(self, "trigger_source", "").lower() in (
            "internal trigger", "software", "software trigger"
        ):
            self.send_trigger()

        # wait for a frame ----------------------------------------------------
        t0 = time.time()
        while not self.frame_buffer:
            if time.time() - t0 > timeout:
                return (None, None) if returnFrameNumber else None
            if self.lastFrameFromBuffer is not None: # in case we are in trigger mode
                return (self.lastFrameFromBuffer, self.lastFrameId) if returnFrameNumber else self.lastFrameFromBuffer
            time.sleep(0.005)

        # Get the latest frame and remove it from the buffer to avoid repeated consumption
        latest_frame = self.frame_buffer[-1] # TODO: Does this cause any problems? if we do .pop() we remove the frames from the 
        latest_frame_id = self.frameid_buffer[-1] 

        # Store as last frame from buffer for fallback scenarios
        self.lastFrameFromBuffer = latest_frame
        self.lastFrameId = latest_frame_id

        if returnFrameNumber:
            return latest_frame, latest_frame_id
        return latest_frame

    def flushBuffer(self):
        self.frameid_buffer.clear()
        self.frame_buffer.clear()

    def getLastChunk(self):
        """Return *and clear* the entire ring‑buffer as a numpy stack."""
        frames = list(self.frame_buffer)
        ids    = list(self.frameid_buffer)
        self.flushBuffer()

        self.lastFrameFromBuffer = frames[-1] if frames else None
        return np.array(frames), np.array(ids)

    def setROI(self,hpos=None,vpos=None,hsize=None,vsize=None):

        if hsize is not None:
            try:
                c_width = self.camera.MV_CC_SetIntValue("Width",int(hsize))
                self.ROI_width = hsize
                self.__logger.debug(f"Width set to {self.ROI_width}")
            except Exception as e:
                self.__logger.error(e)
                self.__logger.debug("Width is not implemented or not writable")

        if vsize is not None:
            try:
                c_height = self.camera.MV_CC_SetIntValue("Height",int(vsize))
                self.ROI_height = vsize
                self.__logger.debug(f"Height set to {self.ROI_height}")
            except Exception as e:
                self.__logger.error(e)
                self.__logger.debug("Height is not implemented or not writable")

        if hpos is not None:
            try:
                c_offsetx = self.camera.MV_CC_SetIntValue("OffsetX",int(hpos))
                self.ROI_hpos = hpos
                self.__logger.debug(f"OffsetX set to {self.ROI_hpos}")
            except Exception as e:
                self.__logger.error(e)
                self.__logger.debug("OffsetX is not implemented or not writable")

        if vpos is not None:
            try:
                c_offsety = self.camera.MV_CC_SetIntValue("OffsetY",int(vpos))
                self.ROI_vpos = vpos
                self.__logger.debug(f"OffsetY set to {self.ROI_vpos}")
            except Exception as e:
                self.__logger.error(e)
                self.__logger.debug("OffsetY is not implemented or not writable")

        return hpos,vpos,hsize,vsize

    def setPropertyValue(self, property_name, property_value):
        if property_name == "gain":
            self.set_gain(property_value)
        elif property_name == "exposure" or property_name == "exposureTime":
            self.set_exposure_time(property_value)
        elif property_name == "exposure_mode":
            self.set_exposure_mode(property_value)
        elif property_name == "blacklevel":
            self.set_blacklevel(property_value)
        elif property_name == "roi_size":
            self.roi_size = property_value
        elif property_name == "frame_rate":
            self.set_frame_rate(property_value)
        elif property_name == "flat_fielding":
            self.set_flatfielding(property_value)
        elif property_name == "trigger_source":
            self.setTriggerSource(property_value)
        elif property_name == "sensor_shutter_mode":
            return self.setSensorShutterMode(property_value)
        elif property_name == 'mode':
            self.set_camera_mode(isAutomatic=property_value)
        else:
            self.__logger.warning(f'Property {property_name} does not exist')
            return False
        return property_value

    # GenICam node types per property. Reading a float node with
    # MV_CC_GetEnumValue does not fail loudly, it just yields 0 - which is why
    # a check of "is the exposure still what I set?" used to answer 0 ms.
    _FLOAT_PROPERTIES = {"gain": "Gain", "exposure": "ExposureTime"}
    _INT_PROPERTIES = {
        "blacklevel": "BlackLevel",
        "image_width": "Width",
        "image_height": "Height",
    }
    _ENUM_PROPERTIES = {"exposure_mode": "ExposureAuto"}

    def getPropertyValue(self, property_name):
        if property_name in self._FLOAT_PROPERTIES:
            node = self._FLOAT_PROPERTIES[property_name]
            value = MVCC_FLOATVALUE()
            ret = self.camera.MV_CC_GetFloatValue(node, value)
            if ret != 0:
                self.__logger.warning(f"Reading {node} failed (0x{ret & 0xffffffff:x})")
                return False
            # ExposureTime is microseconds in the SDK, milliseconds everywhere
            # else in ImSwitch - set_exposure_time multiplies by 1000, so the
            # readback has to divide again.
            return value.fCurValue / 1000 if property_name == "exposure" else value.fCurValue

        if property_name in self._INT_PROPERTIES:
            node = self._INT_PROPERTIES[property_name]
            value = MVCC_INTVALUE()
            ret = self.camera.MV_CC_GetIntValue(node, value)
            if ret != 0:
                self.__logger.warning(f"Reading {node} failed (0x{ret & 0xffffffff:x})")
                return False
            return value.nCurValue

        if property_name in self._ENUM_PROPERTIES:
            node = self._ENUM_PROPERTIES[property_name]
            value = MVCC_ENUMVALUE()
            ret = self.camera.MV_CC_GetEnumValue(node, value)
            if ret != 0:
                self.__logger.warning(f"Reading {node} failed (0x{ret & 0xffffffff:x})")
                return False
            return value.nCurValue

        if property_name == "roi_size":
            return self.roi_size
        if property_name == "frame_Rate":
            return self.frame_rate
        if property_name == "trigger_source":
            return self.trigger_source

        self.__logger.warning(f'Property {property_name} does not exist')
        return False

    def getDeviceModelName(self) -> str:
        """Return the Hikvision device model string (e.g. 'MV-CE050-30GM')."""
        try:
            name = MVCC_STRINGVALUE()
            ret = self.camera.MV_CC_GetStringValue("DeviceModelName", name)
            if ret != 0:
                return ""
            return name.chCurValue.decode(errors="ignore").strip()
        except Exception as e:
            self.__logger.debug(f"getDeviceModelName failed: {e}")
            return ""

    def getSensorShutterMode(self) -> str:
        """Read current SensorShutterMode as string, or '' if unsupported."""
        try:
            stValue = MVCC_ENUMVALUE()
            ret = self.camera.MV_CC_GetEnumValue("SensorShutterMode", stValue)
            if ret != 0:
                return ""
            code = int(stValue.nCurValue)
            # SDK enum codes (from MvCameraControlDefines.h):
            # 0 = Global, 1 = Rolling, 2 = GlobalReset
            return {0: "Global", 1: "Rolling", 2: "GlobalReset"}.get(code, f"code={code}")
        except Exception as e:
            self.__logger.debug(f"getSensorShutterMode failed: {e}")
            return ""

    def setSensorShutterMode(self, mode: str) -> bool:
        """
        Try to set the sensor shutter mode. Returns True on success.

        Many Hikvision sensors don't expose SensorShutterMode at all — some
        CE-series rolling-shutter models are fixed in hardware, and most
        CS-series global-shutter sensors (e.g. MV-CS013-60GN with IMX287)
        are also fixed. In both cases the parameter simply does not exist
        on the camera, and writing to it can destabilise the SDK state.

        This method probes with GetEnumValue first — if the read fails,
        the parameter is unsupported and we return False WITHOUT touching
        the live stream or attempting the write. If the parameter exists
        and is already at the requested mode, we return True as a no-op.

        Valid modes: 'Global', 'Rolling', 'GlobalReset'.
        """
        wanted = str(mode).strip()
        canonical = wanted.replace(" ", "").replace("_", "").lower()
        mode_map = {"global": "Global", "rolling": "Rolling", "globalreset": "GlobalReset"}
        if canonical not in mode_map:
            self.__logger.warning(f"Unknown shutter mode requested: {mode}")
            return False
        target = mode_map[canonical]

        model = self.getDeviceModelName()

        # Probe whether the camera even exposes SensorShutterMode. If not,
        # bail out cleanly — do NOT touch the stream or call SetEnumValue,
        # which can otherwise leave the SDK in a bad state and crash the
        # process on the next MV_CC_* call.
        probe = MVCC_ENUMVALUE()
        probe_ret = self.camera.MV_CC_GetEnumValue("SensorShutterMode", probe)
        if probe_ret != 0:
            self.__logger.info(
                f"SensorShutterMode not supported on '{model}' "
                f"(GetEnumValue ret=0x{probe_ret:x}). "
                "Sensor has a fixed shutter type — no action taken."
            )
            return False

        # Parameter exists. Check current value; skip write if already correct.
        current_before = self.getSensorShutterMode()
        if current_before == target:
            self.__logger.info(
                f"SensorShutterMode already '{target}' on '{model}' — no change needed."
            )
            return True

        was_streaming = self.is_streaming
        if was_streaming:
            self.suspend_live()
        try:
            ret = self.camera.MV_CC_SetEnumValueByString("SensorShutterMode", target)
            if ret != 0:
                self.__logger.warning(
                    f"SensorShutterMode='{target}' rejected by camera "
                    f"(model='{model}', ret=0x{ret:x}). "
                    "Target mode not supported on this sensor."
                )
                return False
            current = self.getSensorShutterMode()
            self.__logger.info(
                f"SensorShutterMode set to '{target}' on '{model}' "
                f"(readback='{current}')"
            )
            return True
        except Exception as e:
            self.__logger.warning(f"setSensorShutterMode('{target}') raised: {e}")
            return False
        finally:
            if was_streaming:
                try:
                    self.start_live()
                except Exception as e:
                    self.__logger.error(f"Failed to restart live view after shutter set: {e}")

    def setTriggerSource(self, trigger_source):
        """
        Continous          → free-run
        Internal trigger   → software (SDK) trigger
        External trigger   → hardware trigger on LINE0
        """
        was_streaming = self.is_streaming
        if was_streaming:
            self.suspend_live()                     # safe action (stop grabbing)

        tlow = str(trigger_source).lower()
        try:
            if tlow.find("cont")>=0:
                self.camera.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
                self.__logger.info("Trigger source set to continuous (free run)")
            elif tlow.find("soft")>=0 or tlow in ("internal trigger", "software", "software trigger"):
                ret = self.camera.MV_CC_SetEnumValue("TriggerMode",  MV_TRIGGER_MODE_ON)
                if ret != 0:
                    self.__logger.error(f"Set TriggerMode failed! ret [0x{ret:x}]")
                    return False
                ret = self.camera.MV_CC_SetEnumValue("TriggerSource", MV_TRIGGER_SOURCE_SOFTWARE)
                if ret != 0:
                    self.__logger.error(f"Set TriggerSource failed! ret [0x{ret:x}]")
                    return False
                self.__logger.info("Trigger source set to software trigger")
            elif tlow.find("ext")>=0 or tlow in ("external trigger", "hardware", "line0"):
                self.camera.MV_CC_SetEnumValue("TriggerMode",  MV_TRIGGER_MODE_ON)
                self.camera.MV_CC_SetEnumValue("TriggerSource", MV_TRIGGER_SOURCE_LINE0)
                self.__logger.info("Trigger source set to external trigger (LINE0)")

            else:
                self.__logger.warning(f"Unknown trigger source: {trigger_source}")
                return False

            self.trigger_source = trigger_source      # remember selection
            return True

        finally:
            if was_streaming:                         # resume if we had been running
                self.start_live()

    def send_trigger(self):
        """Fire one software trigger pulse when trigger source is set to software."""
        ret = self.camera.MV_CC_SetCommandValue("TriggerSoftware")
        if ret != 0:
            self.__logger.error(f"Software trigger failed! ret [0x{ret:x}]")
            return False
        return True

    def openPropertiesGUI(self):
        pass

    def recordFlatfieldImage(self, nFrames=10, nGauss=5, nMedian=5):
        for iFrame in range(nFrames):
            frame = self.getLast()
            if frame is None:
                continue
            if iFrame == 0:
                flatfield = frame
            else:
                flatfield += frame
        flatfield = flatfield / nFrames
        flatfield = gaussian(flatfield, sigma=nGauss)
        flatfield = median(flatfield, selem=np.ones((nMedian, nMedian)))
        self.flatfieldImage = flatfield

    def getFrameNumber(self):
        return self.frameNumber

    # ── helper ---------------------------------------------------------------
    def _hw_timestamp(self, info):
        """Return 64-bit device time-stamp from MV_FRAME_OUT_INFO_EX."""
        try:                     # new SDK (≥2019)
            hi = info.nDevTimeStampHigh
            lo = info.nDevTimeStampLow
            return (hi << 32) | lo
        except AttributeError:   # very old SDK
            return getattr(info, "nHostTimeStamp", 0)
# ----------------------------------------------------------------------------
# Convenience: context‑manager support
# ----------------------------------------------------------------------------
    def __enter__(self):
        self.start_live()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
