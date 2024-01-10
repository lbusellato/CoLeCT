import argparse
import ctypes
import ntplib
import os
import piexif
import time

from os.path import join, dirname, abspath
from PIL import Image

ROOT = dirname(dirname(abspath(__file__)))

libcast_handle = ctypes.CDLL("./libcast.so", ctypes.RTLD_GLOBAL)._handle  # load the libcast.so shared library
pyclariuscast = ctypes.cdll.LoadLibrary("./pyclariuscast.so")  # load the pyclariuscast.so shared library
import pyclariuscast

# Prepare the image folder
recording_path = join(ROOT, 'us_probe_recordings')
recording_dirname = "recording_" + time.strftime("%Y%m%d-%H%M%S")
recording_dir = join(recording_path, recording_dirname)
os.makedirs(recording_dir)

class Timing():
    def __init__(self) -> None:
        self.time = None
        self.time = self.get_timestamp()
        self.prev_time = self.time
    
    def get_timestamp(self) -> float:
        """Generate a timestamp. Query the NTP server to get the current time, then add the increment
        since last time this function was called.

        Returns
        -------
        float
            The current timestamp.
        """
        if self.time is None:

            NTP_SERVER = 'dk.pool.ntp.org'

            client = ntplib.NTPClient()
            response = client.request(NTP_SERVER, version=3)
            return response.tx_time

        curr_time = time.time()
        increment = curr_time - self.prev_time
        self.prev_time = curr_time
        self.time += increment

        return self.time

global img_count
img_count = 0

timing = Timing()

## called when a new processed image is streamed
# @param image the scan-converted image data
# @param width width of the image in pixels
# @param height height of the image in pixels
# @param sz full size of image
# @param micronsPerPixel microns per pixel
# @param timestamp the image timestamp in nanoseconds
# @param angle acquisition angle for volumetric data
# @param imu inertial data tagged with the frame
def newProcessedImage(image, width, height, sz, micronsPerPixel, timestamp, angle, imu):
    bpp = sz / (width * height)
    #print(
    #    "image: {0}, {1}x{2} @ {3} bpp, {4:.2f} um/px, imu: {5} pts".format(
    #        timestamp, width, height, bpp, micronsPerPixel, len(imu)
    #    ),
    #    end="\r",
    #)
    if bpp == 4:
        img = Image.frombytes("RGBA", (width, height), image)
    else:
        img = Image.frombytes("L", (width, height), image)
    global img_count
    # Encode the timestamp in the UserComment EXIF tag of the image
    timestamp = timing.get_timestamp()
    exif_ifd = {piexif.ExifIFD.UserComment: f"{timestamp}".encode()}
    exif_dict = {"Exif": exif_ifd}
    exif_dat = piexif.dump(exif_dict)

    img.save(join(recording_dir, f"{img_count:05d}.png"), exif=exif_dat)
    img_count += 1
    return


## called when a new raw image is streamed
# @param image the raw pre scan-converted image data, uncompressed 8-bit or jpeg compressed
# @param lines number of lines in the data
# @param samples number of samples in the data
# @param bps bits per sample
# @param axial microns per sample
# @param lateral microns per line
# @param timestamp the image timestamp in nanoseconds
# @param jpg jpeg compression size if the data is in jpeg format
# @param rf flag for if the image received is radiofrequency data
# @param angle acquisition angle for volumetric data
def newRawImage(image, lines, samples, bps, axial, lateral, timestamp, jpg, rf, angle):
    # check the rf flag for radiofrequency data vs raw grey grayscale
    # raw grayscale data is non scan-converted and in polar co-ordinates
    # print(
    #    "raw image: {0}, {1}x{2} @ {3} bps, {4:.2f} um/s, {5:.2f} um/l, rf: {6}".format(
    #        timestamp, lines, samples, bps, axial, lateral, rf
    #    ), end = "\r"
    # )
    # if jpg == 0:
    #    img = Image.frombytes("L", (samples, lines), image, "raw")
    # else:
    #    # note! this probably won't work unless a proper decoder is written
    #    img = Image.frombytes("L", (samples, lines), image, "jpg")
    # img.save("raw_image.jpg")
    return


## called when a new spectrum image is streamed
# @param image the spectral image
# @param lines number of lines in the spectrum
# @param samples number of samples per line
# @param bps bits per sample
# @param period line repetition period of spectrum
# @param micronsPerSample microns per sample for an m spectrum
# @param velocityPerSample velocity per sample for a pw spectrum
# @param pw flag that is true for a pw spectrum, false for an m spectrum
def newSpectrumImage(image, lines, samples, bps, period, micronsPerSample, velocityPerSample, pw):
    return


## called when freeze state changes
# @param frozen the freeze state
def freezeFn(frozen):
    if frozen:
        print("\nimaging frozen")
    else:
        print("imaging running")
    return


## called when a button is pressed
# @param button the button that was pressed
# @param clicks number of clicks performed
def buttonsFn(button, clicks):
    print("button pressed: {0}, clicks: {1}".format(button, clicks))
    return

## main function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", "-a", dest="ip", help="ip address of probe.")
    parser.add_argument("--port", "-p", dest="port", type=int, help="port of the probe")
    parser.add_argument("--width", "-w", dest="width", type=int, help="image output width in pixels")
    parser.add_argument("--height", "-ht", dest="height", type=int, help="image output height in pixels")
    parser.set_defaults(ip="192.168.1.1")
    parser.set_defaults(port=5828)
    parser.set_defaults(width=640)
    parser.set_defaults(height=480)
    args = parser.parse_args()

    # get home path
    path = os.path.expanduser("~/")

    # initialize
    cast = pyclariuscast.Caster(newProcessedImage, newRawImage, newSpectrumImage, freezeFn, buttonsFn)
    ret = cast.init(path, args.width, args.height)
    if ret:
        print("initialization succeeded")
        ret = cast.connect(args.ip, args.port, "research")
        if ret:
            print("connected to {0} on port {1}".format(args.ip, args.port))
        else:
            print("connection failed")
            # unload the shared library before destroying the cast object
            ctypes.CDLL("libc.so.6").dlclose(libcast_handle)
            cast.destroy()
            return
    else:
        print("initialization failed")
        return

    # input loop
    key = ""
    while key != "q" and key != "Q":
        key = input("press ('q' to quit) ('a' for action): ")
        if key == "a" or key == "A":
            key = input("(f)->freeze, (i)->image, (c)->cine, (d/D)->depth, (g/G)->gain: ")
            if key == "f" or key == "F":
                cast.userFunction(1, 0)
            elif key == "i" or key == "I":
                cast.userFunction(2, 0)
            elif key == "c" or key == "C":
                cast.userFunction(3, 0)
            elif key == "d":
                cast.userFunction(4, 0)
            elif key == "D":
                cast.userFunction(5, 0)
            elif key == "g":
                cast.userFunction(6, 0)
            elif key == "G":
                cast.userFunction(7, 0)
        elif key == "d" or key == "D":
            ret = cast.disconnect()
            if ret:
                print("successful disconnect")
            else:
                print("disconnection failed")

    cast.destroy()


if __name__ == "__main__":
    main()
