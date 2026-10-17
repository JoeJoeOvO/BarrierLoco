"""Playback-only measurements: skill selection is distinct from physical takeoff."""

import json


class JumpProbe:
    def __init__(self, env):
        self.env = env
        self.sensor = env.scene["contact_forces"]
        self.feet, _ = self.sensor.find_bodies(".*_foot")
        self.active = {}
        self.selected = 0
        self.takeoffs = 0
        self.landed = 0

    def emit(self, event, values):
        print("JUMP_PROBE " + json.dumps({"event": event, **values}), flush=True)

    def begin(self, ids, commands, goal, step):
        data = self.env.scene["robot"].data
        for index, command in zip(ids.tolist(), commands.tolist()):
            self.selected += 1
            record = dict(
                attempt=self.selected,
                env=index,
                step=step,
                start=data.root_pos_w[index].tolist(),
                velocity=data.root_lin_vel_b[index].tolist(),
                goal=goal[index].tolist(),
                command=command[:4],
                support=bool((self.sensor.data.net_forces_w[index, self.feet].norm(dim=-1) > 1).any()),
                rise=0.0,
                air_time=0.0,
                takeoff=False,
                landed=False,
            )
            self.active[index] = record
            self.emit("selected", record.copy())

    def tick(self):
        data = self.env.scene["robot"].data
        for index, record in self.active.items():
            contacts = self.sensor.data.net_forces_w[index, self.feet].norm(dim=-1) > 1
            record["support"] |= bool(contacts.any())
            rise = float(data.root_pos_w[index, 2]) - record["start"][2]
            record["rise"] = max(record["rise"], rise)
            record["air_time"] = 0.0 if contacts.any() else record["air_time"] + self.env.step_dt
            # Falling into a trench is not a successful launch. Require support,
            # upward motion, body rise and sustained all-feet-off contact.
            if (
                not record["takeoff"]
                and record["support"]
                and record["air_time"] >= 0.08
                and rise > 0.025
                and float(data.root_lin_vel_w[index, 2]) > 0.2
            ):
                record["takeoff"] = True
                self.takeoffs += 1
                self.emit("upward_takeoff", record.copy())
            if record["takeoff"] and not record["landed"] and contacts.any():
                record["landed"] = True
                self.landed += 1
                self.emit("contact_after_flight", record.copy())

    def finish(self, timer, done):
        for index in list(self.active):
            if timer[index] <= 0 or done[index]:
                record = self.active.pop(index)
                record["failed"] = bool(done[index])
                self.emit("finished", record)
