from enum import Enum


class CamName(Enum):
    """Names of cameras available in the dataset.

    Port is left, starboard is right. (Mnemonic: alphabetical from left to right ('p' < 's').)
    Congratulations, you are now a fully certified boat captain. Please enter your address to receive your boating
    license in the mail.

    Note: the numbers refer roughly to the orientation of each camera (12 o'clock = front, 3 o'clock = right, etc.). Of
    course, please refer to the calibration files for the exact orientation of each camera.
    """
    STARBOARD_FRONT_WIDE = "hdcam_02_starboard_front_roof_wide"
    STARBOARD_REAR_WIDE = "hdcam_04_starboard_rear_roof_wide"
    PORT_REAR_WIDE = "hdcam_08_port_rear_roof_wide"
    PORT_FRONT_WIDE = "hdcam_10_port_front_roof_wide"
    MIDDLE_FRONT_NARROW_LEFT = "hdcam_12_middle_front_roof_narrow_left"
    MIDDLE_FRONT_NARROW_RIGHT = "hdcam_12_middle_front_roof_narrow_right"
    MIDDLE_FRONT_WIDE = "hdcam_12_middle_front_roof_wide"

