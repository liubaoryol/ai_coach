import time
import heapq


class CTimingEvent:
    def __init__(self, time=0, name="", tag="", callback=None):
        self.set_time = time
        self.name = name
        self.tag = tag
        self.callback = callback

    def __gt__(self, other):
        if self.set_time > other.set_time:
            return True
        elif self.set_time < other.set_time:
            return False
        elif self.name > other.name:
            return True
        elif self.name < other.name:
            return False
        elif self.tag > other.tag:
            return True
        elif self.tag < other.tag:
            return False
        else:  # cannot decide
            return False


class CTimerHelper:
    def __init__(self):
        self.timers = []
        self.start_time = 0

    def reset(self):
        self.timers.clear()
        self.start_time = 0

    def start(self):
        pass

    def get_start_time(self):
        return self.start_time

    def get_current_time(self):
        pass

    def add_timer(self, timing_event):
        heapq.heappush(self.timers, timing_event)

    def ringing(self):
        pass

    def do_event(self, changed_data=None):
        pop_data = heapq.heappop(self.timers)
        pop_data.callback(pop_data.set_time, pop_data.name, changed_data)

    def remove_timer(self, obj_name, tag=""):
        del_idx = []
        for idx, tm in enumerate(self.timers):
            if tm.name == obj_name and (tag == "" or tm.tag == tag):
                del_idx.append(idx)
        self.timers = [tm for idx, tm in enumerate(self.timers)
                       if idx not in del_idx]
        heapq.heapify(self.timers)

    def get_timer(self, obj_name, tag):
        for tm in self.timers:
            if tm.name == obj_name and tm.tag == tag:
                return tm
        return None


class CContinuousTimer(CTimerHelper):
    def __init__(self, unit_time, tolerance=0.0):
        super().__init__()
        self.unit_time = unit_time
        self.tol = tolerance

    def start(self):
        self.start_time = time.time()

    def get_current_time(self):
        return (time.time() - self.get_start_time()) / self.unit_time

    def ringing(self):
        tol = self.tol * self.unit_time  # tolerance
        if len(self.timers) > 0:
            return self.timers[0].set_time - self.get_current_time() <= tol
        return False

    def get_unit_time(self):
        return self.unit_time

    def get_actual_elapsed_time(self):
        return time.time() - self.get_start_time()


class CDiscreteTimer(CTimerHelper):
    def __init__(self):
        super().__init__()
        self.current_time = 0

    def reset(self):
        super().reset()
        self.current_time = 0

    def start(self):
        self.start_time = 0

    def get_current_time(self):
        return self.current_time - self.get_start_time()

    def ringing(self):
        if len(self.timers) > 0:
            return self.timers[0].set_time - self.get_current_time() <= 0
        return False

    def tick(self):
        self.current_time = self.current_time + 1
