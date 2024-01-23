import ntplib
import time

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
