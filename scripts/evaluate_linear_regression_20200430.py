"""
Copyright 2020 Teng Huang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from janos import *
import pandas as pd
import numpy as np
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import time
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
pd.options.mode.chained_assignment = None

"""
load data
"""
# This is the data frame for training the predictive models.
historical_student_data = pd.read_csv("./data/college_student_enroll-s1-1.csv")

# This is information of applicants, whose financial aid is to be determined.
# We will use these numbers (SAT, GPA) later in the objective function.
applications = pd.read_csv("./data/college_applications6000.csv")

"""
set the constant in the model
"""
scholarships = [0, 2.5]  # lower and upper bound if the scholarship
n_simulations = 5
student_sizes = [50, 100, 500, 1000]

"""
pretrained model
"""
# Assign X and y
X = historical_student_data[["SAT", "GPA", "merit"]]
y = historical_student_data[["enroll"]]

# Before training the model, standardize SAT and GPA.
# For convenience, we do not standardize merit.
scaler_sat = StandardScaler().fit(X[["SAT"]])
scaler_gpa = StandardScaler().fit(X[["GPA"]])
X['SAT_scaled'] = scaler_sat.transform(X[['SAT']])
X['GPA_scaled'] = scaler_gpa.transform(X[['GPA']])

# Also, standardize the SAT and GPA in the application data
applications["SAT_scaled"] = scaler_sat.transform(applications[["SAT"]])
applications["GPA_scaled"] = scaler_gpa.transform(applications[["GPA"]])


"""
Prepare the output file
"""
now = datetime.now()
date_time = now.strftime("%H-%M-%S-%Y%m%d")
filename = "20200501_linear_regression_" + date_time + ".txt"
output = open(filename, "w")
#output.write("model_id\t\tstudent_size\t\titeration\t\tgurobi_time\t\tjanos_time\t\tobj_val\n")
output.write("PM\t\tstudent_size\t\tconfiguration\t\titeration\t\tjanos_time\t\tgurobi_time\t\tobj_val\n")
output.close()

"""
Experiments:
"""

#for model_id in range(3):
for model_id in [0]:
    """
    First, train models
    """
    if model_id == 0:
        # train a linear regression model
        my_model = LinearRegression().fit(X[["SAT_scaled", "GPA_scaled", "merit"]], y)
    if model_id == 1:
        # train a logistic regression model
        my_model = LogisticRegression(random_state=0, solver='lbfgs').fit(X[["SAT_scaled", "GPA_scaled", "merit"]], y)
    if model_id == 2:
        # train a small neural network:
        my_model = MLPRegressor(hidden_layer_sizes=[10], random_state=0)  ### TODO: how to link training and optimization!
        my_model.fit(X[["SAT_scaled", "GPA_scaled", "merit"]], y)

    for student_size in student_sizes:
        n_applications = student_size
        BUDGET = int(0.2 * n_applications)

        for iter in range(n_simulations):
            random_sample = applications.sample(student_size, random_state=iter)
            random_sample = random_sample.reset_index()

            m = JModel()

            # Define regular variables
            assign_scholarship = m.add_regular_variables([n_applications], "assign_scholarship")
            for app_index in range(n_applications):
                assign_scholarship[app_index].setContinuousDomain(lower_bound=scholarships[0], upper_bound=scholarships[1])
                assign_scholarship[app_index].setObjectiveCoefficient(0)

            # Define predicted variables
            # First, we need to create structures of predictive models. In this case, we associate such a structure with an existing / pretrained logistic regression model.
            logistic_regression_model = OptimizationPredictiveModel(m, pretrained_model=my_model,
                                                                    feature_names=["SAT_scaled", "GPA_scaled", "merit"])

            # Now, we could define the predicted decision variables and associate them with the predicted model structure.
            enroll_probabilities = m.add_predicted_variables([n_applications], "enroll_probs")
            for app_index in range(n_applications):
                enroll_probabilities[app_index].setObjectiveCoefficient(1)
                mapping_of_vars = {"merit": assign_scholarship[app_index],
                                   "SAT_scaled": random_sample["SAT_scaled"][app_index],
                                   "GPA_scaled": random_sample["GPA_scaled"][app_index]}
                enroll_probabilities[app_index].setPM(logistic_regression_model, mapping_of_vars)

            # Construct constraints
            # \sum_i x_i <= BUDGET
            scholarship_deployed = Expression()

            for app_index in range(n_applications):
                scholarship_deployed.add_term(assign_scholarship[app_index], 1)

            m.add_constraint(scholarship_deployed, "less_equal", BUDGET)

            # solve the model
            m.add_gurobi_param_settings('TimeLimit', 1800)
            m.add_gurobi_param_settings('DUALREDUCTIONS', 0)
            m.add_gurobi_param_settings('MIPGap', 0.001)
            m.add_gurobi_param_settings('Threads', 1)
            m.set_output_flag(0)
            m.solve()

            """
            write output
            borrowed from https://www.gurobi.com/documentation/8.1/examples/workforce1_py.html
            """
            status = m.gurobi_model.status

            if status == GRB.Status.UNBOUNDED:
                print('The model cannot be solved because it is unbounded')
                sys.exit(0)
            elif status == GRB.Status.OPTIMAL:
                output = open(filename, "a")
                output.write("LinReg\t\t" + str(student_size) + "\t\tNULL\t\t" + str(iter) + "\t\t" + str(m.get_time()) + "\t\t" + str(m.gurobi_model.runtime) + "\t\t" + str(m.gurobi_model.objval) + "\n")
#                output.write("PM\t\tstudent_size\t\tconfiguration\t\titeration\t\tjanos_time\t\tgurobi_time\t\tobj_val\n")
                output.close()
            elif status != GRB.Status.INF_OR_UNBD and status != GRB.Status.INFEASIBLE:
                print('Optimization was stopped with status %d' % status)
            else:
                # if none of the above, then do IIS
                print('The model is infeasible; computing IIS')
                m.gurobi_model.computeIIS()
                m.gurobi_model.write("ip_model_inf.ilp")
                if m.gurobi_model.IISMinimal:
                    print('IIS is minimal\n')
                else:
                    print('IIS is not minimal\n')
                print('\nThe following constraint(s) cannot be satisfied:')
                for c in m.gurobi_model.getConstrs():
                    if c.IISConstr:
                        print('%s' % c.constrName)
