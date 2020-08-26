"""
Copyright © 2020 Teng Huang
Some rights reserved.
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
scholarships = [0, 0.5, 1.0, 1.5, 2.0, 2.5]
n_simulations = 5
student_sizes = [500, 1000]
interview_sizes = [20]
LAYERS = 1
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

now = datetime.now()
date_time = now.strftime("%H-%M-%S-%Y%m%d")
filename = "rewrite_08_s1_full_" + date_time + ".txt"
output = open(filename, "w")
output.write("Algorithm\tPModel\tn_students\titeration\tobj_val\truntime\n")
output.close()

for model_id in range(3):
    # if model_id != 1:
    #     continue
    if model_id == 0:
        continue

    if model_id == 0:
        my_logistic_regression = LinearRegression().fit(X[["SAT_scaled", "GPA_scaled", "merit"]], y)
    if model_id == 1:
        my_logistic_regression = LogisticRegression(random_state=0, solver='lbfgs').fit(
            X[["SAT_scaled", "GPA_scaled", "merit"]], y)
    if model_id == 2:
        my_logistic_regression = MLPRegressor(hidden_layer_sizes=[10], random_state=0)
        my_logistic_regression.fit(X[["SAT_scaled", "GPA_scaled", "merit"]], y)

    if model_id == 0:
        model_name = "LinReg"
    if model_id == 1:
        model_name = "LogReg"
    if model_id == 2:
        model_name = "NN"

    for n_students in student_sizes:
        n_applications = n_students
        n_administration_letters = n_students
        BUDGET = int(0.2 * n_students)

        for sim_idx in range(n_simulations):

            # randomly select n_administration_letters samples.
            random_sample = applications.sample(n_administration_letters, random_state=sim_idx)
            random_sample = random_sample.reset_index()

            random_sample["no_merit"] = [scholarships[0]] * n_applications
            random_sample["yes_merit"] = [scholarships[-1]] * n_applications

            if model_id == 1:
                predicted_probabilities = my_logistic_regression.predict_proba(
                    random_sample[["SAT_scaled", "GPA_scaled", "no_merit"]])
                predicted_probabilities = pd.DataFrame(predicted_probabilities, columns=["0", '1'])
                random_sample["enroll_probability_no_merit"] = predicted_probabilities["1"]

                predicted_probabilities = my_logistic_regression.predict_proba(
                    random_sample[["SAT_scaled", "GPA_scaled", "yes_merit"]])
                predicted_probabilities = pd.DataFrame(predicted_probabilities, columns=["0", "1"])
                random_sample["enroll_probability_yes_merit"] = predicted_probabilities["1"]
            else:
                predicted_probabilities = my_logistic_regression.predict(
                    random_sample[["SAT_scaled", "GPA_scaled", "no_merit"]])
                # predicted_probabilities = pd.DataFrame(predicted_probabilities, columns=["0", '1'])
                random_sample["enroll_probability_no_merit"] = predicted_probabilities

                predicted_probabilities = my_logistic_regression.predict(
                    random_sample[["SAT_scaled", "GPA_scaled", "yes_merit"]])
                # predicted_probabilities = pd.DataFrame(predicted_probabilities, columns=["0", "1"])
                random_sample["enroll_probability_yes_merit"] = predicted_probabilities

            random_sample["enroll_probability_diff"] = random_sample['enroll_probability_yes_merit'] - random_sample[
                'enroll_probability_no_merit']

            """
            non-greedy heuristic (Teng)
            """
            # Sort by enroll_probability_yes_merit
            baseline_start_time = time.time()
            random_sample = random_sample.sort_values(by=['enroll_probability_yes_merit'], ascending=False)
            random_sample = random_sample.reset_index(drop=True)

            obj_val = 0.0
            for i in range(n_administration_letters):
                if i < int(BUDGET / scholarships[-1]):
                    probability = random_sample['enroll_probability_yes_merit'][i]
                else:
                    probability = random_sample['enroll_probability_no_merit'][i]
                obj_val += probability
            baseline_end_time = time.time()
            total_time = baseline_end_time - baseline_end_time
            output = open(filename, "a")
            output.write("non-greedy\t" + model_name + "\t" + str(n_students) + "\t" + str(sim_idx) + "\t" + str(
                obj_val) + "\t" + str(total_time) + "\n")
            output.close()
            """
            JANOS: predict and prescribe (discrete)
            """
            m = JModel()

            # Define regular variables
            assign_scholarship = m.add_regular_variables([n_administration_letters], "assign_scholarship")
            for app_index in range(n_administration_letters):
                assign_scholarship[app_index].setDiscreteDomain(scholarships)
                assign_scholarship[app_index].setObjectiveCoefficient(0)

            # Define predicted variables
            # First, we need to create structures of predictive models. In this case, we associate such a structure with an existing / pretrained logistic regression model.

            logistic_regression_model = OptimizationPredictiveModel(m, pretrained_model=my_logistic_regression,
                                                                    feature_names=["SAT_scaled", "GPA_scaled", "merit"])

            # Now, we could define the predicted decision variables and associate them with the predicted model structure.
            enroll_probabilities = m.add_predicted_variables([n_administration_letters], "enroll_probs")
            for app_index in range(n_administration_letters):
                enroll_probabilities[app_index].setObjectiveCoefficient(1)
                mapping_of_vars = {"merit": assign_scholarship[app_index],
                                   "SAT_scaled": random_sample["SAT_scaled"][app_index],
                                   "GPA_scaled": random_sample["GPA_scaled"][app_index]}
                enroll_probabilities[app_index].setPM(logistic_regression_model, mapping_of_vars)

            # Construct constraints
            # \sum_i x_i <= BUDGET
            scholarship_deployed = Expression()

            for app_index in range(n_administration_letters):
                scholarship_deployed.add_term(assign_scholarship[app_index], 1)

            m.add_constraint(scholarship_deployed, "less_equal", BUDGET)

            # solve the model
            m.add_gurobi_param_settings('TimeLimit', 1800)
            m.add_gurobi_param_settings('DUALREDUCTIONS', 0)
            m.add_gurobi_param_settings('MIPGap', 0.001)
            m.add_gurobi_param_settings('Threads', 1)
            m.set_output_flag(0)
            m.solve()

            status = m.gurobi_model.status
            output = open(filename, "a")
            output.write("janos_discrete\t" + model_name + "\t" + str(n_students) + "\t" + str(sim_idx) + "\t" + str(
                m.gurobi_model.objBound) + "\t" + str(m.get_time()) + "\t" + str(status) + "\n")
            output.close()
            #            if status == GRB.Status.UNBOUNDED:
            #                print('The model cannot be solved because it is unbounded')
            #                sys.exit(0)
            #            elif status == GRB.Status.OPTIMAL:
            #                output.write("janos_discrete\t" + model_name + "\t" + str(n_students) + "\t" + str(sim_idx) + "\t" + str(m.gurobi_model.objVal) + "\t" + str(m.get_time()) + "\n")
            #
            #            elif status != GRB.Status.INF_OR_UNBD and status != GRB.Status.INFEASIBLE:
            #                print('Optimization was stopped with status %d' % status)
            #            else:
            #                # if none of the above, then do IIS
            #                print('The model is infeasible; computing IIS')
            #                m.gurobi_model.computeIIS()
            #                m.gurobi_model.write("ip_model_inf.ilp")
            #                if m.gurobi_model.IISMinimal:
            #                    print('IIS is minimal\n')
            #                else:
            #                    print('IIS is not minimal\n')
            #                print('\nThe following constraint(s) cannot be satisfied:')
            #                for c in m.gurobi_model.getConstrs():
            #                    if c.IISConstr:
            #                        print('%s' % c.constrName)
            """
            greedy heuristic (David)
            """
            # sort by enroll_probability_diff
            baseline_start_time = time.time()
            random_sample = random_sample.sort_values(by=['enroll_probability_diff'], ascending=False)
            random_sample = random_sample.reset_index(drop=True)

            obj_val = 0.0
            for i in range(n_administration_letters):
                if i < int(BUDGET / scholarships[-1]):
                    probability = random_sample['enroll_probability_yes_merit'][i]
                else:
                    probability = random_sample['enroll_probability_no_merit'][i]
                obj_val += probability
            baseline_end_time = time.time()
            total_time = baseline_end_time - baseline_end_time
            output = open(filename, "a")
            output.write("greedy\t" + model_name + "\t" + str(n_students) + "\t" + str(sim_idx) + "\t" + str(
                obj_val) + "\t" + str(total_time) + "\n")
            output.close()

            """
            JANOS: predict and prescribe (continuous)
            """
            m = JModel()

            # Define regular variables
            assign_scholarship = m.add_regular_variables([n_administration_letters], "assign_scholarship")
            for app_index in range(n_administration_letters):
                assign_scholarship[app_index].setContinuousDomain(scholarships[0], scholarships[-1])
                assign_scholarship[app_index].setObjectiveCoefficient(0)

            # Define predicted variables
            # First, we need to create structures of predictive models. In this case, we associate such a structure with an existing / pretrained logistic regression model.

            logistic_regression_model = OptimizationPredictiveModel(m, pretrained_model=my_logistic_regression,
                                                                    feature_names=["SAT_scaled", "GPA_scaled", "merit"])

            # Now, we could define the predicted decision variables and associate them with the predicted model structure.
            enroll_probabilities = m.add_predicted_variables([n_administration_letters], "enroll_probs")
            for app_index in range(n_administration_letters):
                enroll_probabilities[app_index].setObjectiveCoefficient(1)
                mapping_of_vars = {"merit": assign_scholarship[app_index],
                                   "SAT_scaled": random_sample["SAT_scaled"][app_index],
                                   "GPA_scaled": random_sample["GPA_scaled"][app_index]}
                enroll_probabilities[app_index].setPM(logistic_regression_model, mapping_of_vars)

            # Construct constraints
            # \sum_i x_i <= BUDGET
            scholarship_deployed = Expression()

            for app_index in range(n_administration_letters):
                scholarship_deployed.add_term(assign_scholarship[app_index], 1)

            m.add_constraint(scholarship_deployed, "less_equal", BUDGET)

            # solve the model
            m.add_gurobi_param_settings('TimeLimit', 1800)
            m.add_gurobi_param_settings('DUALREDUCTIONS', 0)
            m.add_gurobi_param_settings('MIPGap', 0.001)
            m.add_gurobi_param_settings('Threads', 1)
            m.set_output_flag(0)
            m.solve()

            status = m.gurobi_model.status
            output = open(filename, "a")
            output.write("janos_continuous\t" + model_name + "\t" + str(n_students) + "\t" + str(sim_idx) + "\t" + str(
                m.gurobi_model.objBound) + "\t" + str(m.get_time()) + "\t" + str(status) + "\n")
            output.close()

#            if status == GRB.Status.UNBOUNDED:
#                print('The model cannot be solved because it is unbounded')
#                sys.exit(0)
#            elif status == GRB.Status.OPTIMAL:
#                output.write("janos_discrete\t" + model_name + "\t" + str(n_students) + "\t" + str(sim_idx) + "\t" + str(m.gurobi_model.objVal) + "\t" + str(m.get_time()) + "\n")
#
#            elif status != GRB.Status.INF_OR_UNBD and status != GRB.Status.INFEASIBLE:
#                print('Optimization was stopped with status %d' % status)
#            else:
#                # if none of the above, then do IIS
#                print('The model is infeasible; computing IIS')
#                m.gurobi_model.computeIIS()
#                m.gurobi_model.write("ip_model_inf.ilp")
#                if m.gurobi_model.IISMinimal:
#                    print('IIS is minimal\n')
#                else:
#                    print('IIS is not minimal\n')
#                print('\nThe following constraint(s) cannot be satisfied:')
#                for c in m.gurobi_model.getConstrs():
#                    if c.IISConstr:
#                        print('%s' % c.constrName)
#                sys.exit(1)
