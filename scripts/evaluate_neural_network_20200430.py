# 16:10 on April 30, 2020
# Teng Huang
# Description:
# copied from evaluation_neural_network_20200107.py
# for first revision of JANOS paper
# use data -v3 instead of -v2


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
from sklearn.metrics import mean_squared_error

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
n_simulations = 5  # to have meaningful mean and standard deviation;
student_sizes = [50, 100, 500, 1000]  # we measure these predictions' RMSE
# interview_sizes = [5, 10, 15, 20, 25]
LAYERS = 3
nodes_per_layer = 10
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
filename = "20200501_neural_network_" + date_time + ".txt"
output = open(filename, "w")
output.write("PM\t\tstudent_size\t\tn_layers\t\titeration\t\tjanos_time\t\tgurobi_time\t\tobj_val\n")
output.close()

for student_size in student_sizes:
    n_applications = student_size
    BUDGET = int(0.2 * n_applications)
    hidden_layer_sizes = []
    for n_layers in range(LAYERS):

        hidden_layer_sizes.append(nodes_per_layer)

        my_logistic_regression = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes, random_state=0)  ### TODO: how to link training and optimization!
        my_logistic_regression.fit(X[["SAT_scaled", "GPA_scaled", "merit"]], y)

        for iter in range(n_simulations):
            random_sample = applications.sample(student_size, random_state=iter)
            random_sample = random_sample.reset_index()

            m = JModel()

            # Define regular variables
            assign_scholarship = m.add_regular_variables([n_applications], "assign_scholarship")
            for app_index in range(n_applications):
                assign_scholarship[app_index].setContinuousDomain(lower_bound=scholarships[0],
                                                                  upper_bound=scholarships[1])
                assign_scholarship[app_index].setObjectiveCoefficient(0)

            # Define predicted variables
            # First, we need to create structures of predictive models. In this case, we associate such a structure with an existing / pretrained logistic regression model.
            logistic_regression_model = OptimizationPredictiveModel(m, pretrained_model=my_logistic_regression,
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
            #            m.add_gurobi_param_settings("MIPGap", 0.01)

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
                output.write("NN\t\t" + str(student_size) + "\t\t" + str(n_layers) + "\t\t" + str(iter) +
                             "\t\t" + str(m.get_time()) + "\t\t" + str(m.gurobi_model.runtime) +
                             "\t\t" + str(m.gurobi_model.objval) + "\n")
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
