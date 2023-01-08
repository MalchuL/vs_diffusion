from src.models.components.utils.interpolation.tweenings import interpolate


class Interpolator:
    def __init__(self, method, num_steps, inverse=False):
        """
        method - string name of pytweening lib
        num_steps - maximal number of steps
        """
        self.num_steps = num_steps
        self.method = method
        self.inverse = inverse

    def interpolate(self, step):
        if step >= self.num_steps or self.method is None:
            if self.inverse:
                return 0.0
            else:
                return 1.0
        else:
            alpha = max(min(step / self.num_steps, 1.0), 0.0)
            out_alpha = interpolate(alpha, self.method)
            if self.inverse:
                out_alpha = 1 - out_alpha
        return out_alpha


class MultiInterpolator:
    def __init__(self, interpolators):
        self.interpolators = interpolators
        self.milestones = []
        current_milestone = 0
        for interpolator in self.interpolators:
            current_milestone += interpolator.num_steps
            self.milestones.append({'milestone': current_milestone, 'interpolator': interpolator})

    def interpolate(self, step):
        prev_milestone = 0
        for interpolation in self.milestones[:-1]:
            interpolator = interpolation['interpolator']
            milestone = interpolation['milestone']
            if step < milestone:
                return interpolator.interpolate(step - prev_milestone)
            prev_milestone = milestone

        return self.milestones[-1]['interpolator'].interpolate(step - prev_milestone)

