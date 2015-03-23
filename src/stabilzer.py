from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus

__author__ = 'Marek'


class OptimalCameraPathStabilizer:
    def __init__(self, transformations, maximal_windows_shift):
        self.original_transformations = transformations
        self.original_path_length = len(transformations)
        self.maximal_windows_shift = maximal_windows_shift
        self.weights = [1, 10, 100]

        self.e_vars = None
        self.b_vars = None

    def create_lp_problem(self):
        return LpProblem(self.__class__.__name__, LpMinimize)

    def create_variables(self):
        self.create_e_vars()
        self.create_b_vars()

    def create_e_vars(self):
        self.e_vars = [[LpVariable("e%1d_%03d" % (wi, t), lowBound=0) for t in xrange(self.original_path_length)] for wi in
                       xrange(len(self.weights))]

    def create_b_vars(self):
        self.b_vars = [LpVariable("b%03d" % t, lowBound=-self.maximal_windows_shift, upBound=self.maximal_windows_shift)
                       for t in xrange(self.original_path_length)]

    def create_minimization_objective(self):
        objective = []
        for weight_index, weight in enumerate(self.weights):
            objective += [weight * self.e_vars[weight_index][t] for t in xrange(len(self.e_vars[weight_index]))]
        return objective

    def add_constraints_to(self, prob):
        for t in xrange(len(self.e_vars[0])):
            # constant path
            prob += -self.e_vars[0][t] <= self.residual(t) <= self.e_vars[0][t]

            if len(self.weights) <= 1:
                continue

            # constant velocity
            prob += -self.e_vars[1][t] <= self.residual(t) - self.residual(t - 1) <= self.e_vars[1][t]

            if len(self.weights) <= 2:
                continue

            # constant acceleration
            prob += -self.e_vars[2][t] <= self.residual(t) - 2 * self.residual(
                t - 1) + self.residual(t - 2) <= self.e_vars[2][t]

    def residual(self, t):
        f_t = self.original_transformations[t] if t >= 0 else 0
        b_t_min_1 = self.b_vars[t - 1] if t - 1 >= 0 else 0
        b_t = self.b_vars[t] if t >= 0 else 0

        return f_t + b_t - b_t_min_1

    def stabilize(self, ):
        self.create_variables()
        prob = self.create_lp_problem()
        prob += lpSum(self.create_minimization_objective())
        self.add_constraints_to(prob)
        prob.solve()

        return [b.varValue for b in prob.variables() if "b" in b.name], LpStatus[prob.status]

    @staticmethod
    def cleanup():
        import os
        file_list = [f for f in os.listdir(".") if f.endswith(".mps")]
        for f in file_list:
            os.remove(f)
