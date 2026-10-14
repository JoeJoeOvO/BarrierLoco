"""Measured wheel-bottom peaks over complete commanded swing phases."""


class SwingHeightTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.previous = [None] * 4
        self.active_peak = [None] * 4
        self.last_peak = [None] * 4

    def update(self, heights, swinging):
        for index, (height, swing) in enumerate(zip(heights, swinging)):
            if swing:
                if self.previous[index] is False:
                    self.active_peak[index] = float(height)
                elif self.active_peak[index] is not None:
                    self.active_peak[index] = max(self.active_peak[index], float(height))
            elif self.previous[index] is True and self.active_peak[index] is not None:
                self.last_peak[index] = max(self.active_peak[index], float(height))
                self.active_peak[index] = None
            # A run/case beginning mid-swing must wait for a full swing to report a peak.
            self.previous[index] = bool(swing)
        return list(self.last_peak)


def format_heights_cm(values):
    return " / ".join("--" if value is None else f"{100 * value:.1f}" for value in values)
