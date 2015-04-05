import uuid
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus

__author__ = 'Marek'


class OptimalPathStabilizer:
    def __init__(self, transformations, maximal_windows_shift):
        self.original_transformations = transformations
        self.original_path_length = len(transformations)
        self.maximal_windows_shift = maximal_windows_shift
        self.weights = [1, 10, 100]

        self.e_vars = None
        self.b_vars = None

        self.e_name_prefix = "e_"
        self.b_name_prefix = "b_"

        self.residual_factory = self.__class__.simple_1d_residual

    @staticmethod
    def simple_1d_residual(stabilizer, t):
        f_t = stabilizer.original_transformations[t] if t >= 0 else 0
        b_t_min_1 = stabilizer.b_vars[t - 1] if t - 1 >= 0 else 0
        b_t = stabilizer.b_vars[t] if t >= 0 else 0

        return f_t + b_t - b_t_min_1

    @classmethod
    def create_lp_problem(cls):
        return LpProblem(cls.__name__ + str(uuid.uuid4()), LpMinimize)

    def create_variables(self):
        self.create_e_vars()
        self.create_b_vars()

    def create_e_vars(self):
        self.e_vars = []
        for wi in xrange(len(self.weights)):
            e_w = []
            for t in xrange(self.original_path_length):
                e_w.append(LpVariable("%s_%1d_%06d" % (self.e_name_prefix, wi, t), lowBound=0))
            self.e_vars.append(e_w)

    def create_b_vars(self):
        self.b_vars = []
        for t in xrange(self.original_path_length):
            self.b_vars.append(LpVariable("%s_%06d" % (self.b_name_prefix, t), lowBound=-self.maximal_windows_shift, upBound=self.maximal_windows_shift))

    def create_minimization_objective(self):
        objective = []
        for weight_index, weight in enumerate(self.weights):
            objective += [weight * self.e_vars[weight_index][t] for t in xrange(self.original_path_length)]
        return objective

    def add_constraints_to(self, prob):
        for t in xrange(self.original_path_length):
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
        return self.residual_factory(self, t)

    def stabilize(self):
        self.create_variables()
        prob = self.create_lp_problem()
        prob += lpSum(self.create_minimization_objective())
        self.add_constraints_to(prob)

        prob.solve()
        OptimalPathStabilizer.cleanup()

        return [b.varValue for b in prob.variables() if "b" in b.name], LpStatus[prob.status]

    @staticmethod
    def cleanup():
        import os
        file_list = [f for f in os.listdir(".") if f.endswith(".mps")]
        for f in file_list:
            os.remove(f)


class OptimalPathStabilizerXYA:
    def __init__(self, x, y, a, limits):
        self.original_transformations = [x, y, a]
        self.limits = limits
        self.labels = ["x", "y", "a"]
        self.stabilizers = [OptimalPathStabilizer(self.original_transformations[d], limits[d]) for d in xrange(3)]

        for d, stabilizer in enumerate(self.stabilizers):
            stabilizer.e_name_prefix = "e_%s_" % self.labels[d]
            stabilizer.b_name_prefix = "b_%s_" % self.labels[d]
    #         stabilizer.residual_factory = self.class_residual
    #
    # def class_residual(self, stabilizer, t):
    #     # dimension = self.stabilizers.index(stabilizer)
    #
    #     f_t = stabilizer.original_transformations[t] if t >= 0 else 0
    #     b_t_min_1 = stabilizer.b_vars[t - 1] if t - 1 >= 0 else 0
    #     b_t = stabilizer.b_vars[t] if t >= 0 else 0
    #
    #     return f_t + b_t - b_t_min_1

    def stabilize(self):
        for stabilizer in self.stabilizers:
            stabilizer.create_variables()

        prob = OptimalPathStabilizer.create_lp_problem()

        objective = []
        for stabilizer in self.stabilizers:
            objective += [stabilizer.create_minimization_objective()]
        prob += lpSum(objective)

        for stabilizer in self.stabilizers:
            stabilizer.add_constraints_to(prob)

        prob.solve()
        OptimalPathStabilizer.cleanup()

        result = []
        for label, stabilizer in zip(self.labels, self.stabilizers):
            result += [[b.varValue for b in prob.variables() if "b_%s_" % label in b.name]]

        return result