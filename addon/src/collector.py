from context import Context


class Collector:
    def __init__(self, context: Context):
        self.context = context
        pass

    def tick(self):
        pass

    def update_sensors(self):
        #         context.pv_buffer.append(current_pv)
        #         context.stable_pv = np.median(self.pv_buffer)
        #
        #         self.load_buffer.append(current_load)
        #         context.stable_load = np.median(self.load_buffer)
        pass

    def log_snapshot(self):
        pass

    def update_pv(self):
        pass
