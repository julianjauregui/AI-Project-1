#####Importations#####
#these are the following libraries that will be used for the project
#ensure that these packages are installed before hand by using pip install
import csv
from xml.etree.ElementTree import tostring
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import cmocean
import os
import seaborn as sns

#####################



#####Connect to datasets#####

#This folder contains the datasets that we will be using for the project
import Datasets


#Interactions holds the data that happen between the players and the NPCs interactions
interactions = pd.read_csv(os.path.join("Datasets", "Interactions_Final_v10_Dataset.csv"))
#ogNPC holds the information of the original NPCs and their interactions
ogNPC = pd.read_csv(os.path.join("Datasets", "Updated_Old_NPCs.csv"))
#newNPC holds the information of the new, seasonal, NPCs and their interactions
newNPC = pd.read_csv(os.path.join("Datasets", "Updated_Seasonal_NPCs.csv"))

#####################



#####Interactions summary statistics#####

#this line ensures that all columns are shown for the pandas operations and summaries
pd.set_option('display.max_columns', None)

#prints a couple of sample lines and summarizes the interaction dataset
print("Sample Interactions: ")
print(interactions.head())
print("\n\n\n")
print("Interaction Summary: ")
print(interactions.describe())

########################



#####Summary graphs of datasets#####

#This helps us visualize the interactions dataset
#the first category is the interest
plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1)

sns.histplot(data=interactions, x="Interest")
plt.title("Distribution of Interest")
#the second category is the User Level
plt.subplot(2, 3, 2)
sns.histplot(data=interactions, x="User level")
plt.title("Distribution of User Level")

#the third category is the NPC Friendliness
plt.subplot(2, 3, 3)
sns.histplot(data=interactions, x="NPC friendliness")
plt.title("Distribution of NPC Friendliness")

#the fourth category is the Interaction length
plt.subplot(2, 3, 4)
sns.histplot(data=interactions, x="Interaction length")
plt.title("Distribution of Interaction length")

#the fifth category is the Interaction quests acquired
plt.subplot(2, 3, 5)
sns.histplot(data=interactions, x="Interaction quests acquired")
plt.title("Distribution of Interaction quests acquired")
plt.tight_layout()
plt.show()



#This helps us visualize the oldNPC dataset

plt.figure(figsize=(12, 8))
#the first category is the NPC friendliness
plt.subplot(2, 3, 1)
x = ogNPC["NPC friendliness"]
y = newNPC["NPC friendliness"]
plt.hist([x,y], label=['Old NPC', 'New/seasonal NPC'], color=['blue', 'orange'], alpha=0.7)
plt.title("Distribution of NPC friendliness")
plt.legend()
#the second category is the NPC item value
plt.subplot(2, 3, 2)
x = ogNPC["NPC item value"]
y = newNPC["NPC item value"]
plt.hist([x,y], label=['Old NPC', 'New/seasonal NPC'], color=['blue', 'orange'], alpha=0.7)
plt.title("Distribution of NPC item value")
plt.legend()
#the third category is the NPC quest value
plt.subplot(2, 3, 3)
x = ogNPC["NPC quest value"]
y = newNPC["NPC quest value"]
plt.hist([x,y], label=['Old NPC', 'New/seasonal NPC'], color=['blue', 'orange'], alpha=0.7)
plt.title("Distribution of NPC quest value")
plt.legend()
#the fourth category is the NPC gender
plt.subplot(2, 3, 4)
x = ogNPC["NPC gender"]
y = newNPC["NPC gender"]
plt.hist([x,y], label=['Old NPC', 'New/seasonal NPC'], color=['blue', 'orange'], alpha=0.7)
plt.title("Distribution of NPC gender")
plt.legend()
#the fifth category is the NPC class
plt.subplot(2, 3, 5)
x = ogNPC["NPC class"]
y = newNPC["NPC class"]
plt.hist([x,y], label=['Old NPC', 'New/seasonal NPC'], color=['blue', 'orange'], alpha=0.7)
plt.title("Distribution of NPC class")
plt.legend()
#the sixth category is the NPC area level
plt.subplot(2, 3, 6)
x = ogNPC["NPC area level"]
y = newNPC["NPC area level"]
plt.hist([x,y], label=['Old NPC', 'New/seasonal NPC'], color=['blue', 'orange'], alpha=0.7)
plt.title("Distribution of NPC area level")
plt.legend()
plt.show()


# First, visualize raw relationships (without scaling)
#We will be making the target the interaction length
# This style of plot uses multiple panels within the same figure
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)

#this maps interaction length vs user level
plt.scatter(interactions["User level"], interactions["Interaction length"], alpha=0.5, color="crimson")
plt.xlabel("User level")
plt.ylabel("Interaction length")
plt.title("Interaction length vs User Level")

#this maps Interaction length vs NPC Friendliness
plt.subplot(2, 2, 2)
plt.scatter(interactions["NPC friendliness"], interactions["Interaction length"], alpha=0.5, color="darkorange")
plt.xlabel("NPC friendliness")
plt.ylabel("Interaction length")
plt.title("Interaction length vs. NPC friendliness")

#this maps Interaction length vs interest
plt.subplot(2, 2, 3)
plt.scatter(interactions["Interest"], interactions["Interaction length"], alpha=0.5, color="royalblue")
plt.xlabel("Interest")
plt.ylabel("Interaction length")
plt.title("Interaction length vs. Interest")

#this maps Interaction length vs interaction quests acquired
plt.subplot(2, 2, 4)
plt.scatter(interactions["Interaction quests acquired"], interactions["Interaction length"], alpha=0.5, color="lightgreen")
plt.xlabel("Interaction quests acquired")
plt.ylabel("Interaction length")
plt.title("Interaction length vs Interaction quests acquired")


plt.tight_layout()
plt.show()

###################



##### Run the training #####
# Your functions here
# Extract relevant features for training
x_train_2 = interactions["Interest"].values.reshape(-1, 1)  # Feature square footage
x_train_5 = interactions["Interaction quests acquired"].values.reshape(-1, 1)  # Feature house age
y_true = interactions["Interaction length"].values.reshape(-1, 1)  # Target (Price)

# Function to predict with the linear model
def predict_multiple(x_train_2, x_train_5, w_2, w_5, b):
    return w_2 * x_train_2 + w_5 * x_train_5 + b

# Function to compute cost (Mean Squared Error)
def compute_cost_multiple(x_train_2, x_train_5, w_2, w_5, b, y_true):
    m = len(y_true)
    y_pred = predict_multiple(x_train_2, x_train_5, w_2, w_5, b)
    return (1 / (2 * m)) * np.sum((y_pred - y_true) ** 2)

# Function to perform one iteration of gradient descent
def multiple_descent_step(x_train_2, x_train_5, w_2, w_5, b, y_true, alpha):
    m = len(y_true)
    y_pred = predict_multiple(x_train_2, x_train_5, w_2, w_5, b)
    dw_2 = (1 / m) * np.sum((y_pred - y_true) * x_train_2)
    dw_5 = (1 / m) * np.sum((y_pred - y_true) * x_train_5)
    db = (1 / m) * np.sum(y_pred - y_true)
    w_2 -= alpha * dw_2
    w_5 -= alpha * dw_2
    b -= alpha * db
    return w_2, w_5, b

# Training function
def train_multiple_regression(x_train_2, x_train_5, w_2, w_5, b, y_true, alpha, iterations):
    cost_history = []
    for i in range(iterations):
        w_2, w_5, b = multiple_descent_step(x_train_2, x_train_5, w_2, w_5, b, y_true, alpha)
        cost_history.append(compute_cost_multiple(x_train_2, x_train_5, w_2, w_5, b, y_true))
    return w_2, w_5, b, cost_history

# Your training code and plot-making here
# Initialize model parameters
w_2 = 0  # Weight (slope)
w_5 = 0
b = 0  # Bias (intercept)

# Train the model
alpha = 0.000485  # Learning rate
iterations = 200
w_optimal_2, w_optimal_5, b_optimal, cost_history = train_multiple_regression(x_train_2, x_train_5, w_2, w_5, b, y_true, alpha, iterations)

#This function is used to predict the line for the next graph, actual points vs predicted line
def predict(x_train_2, x_train_5, w_2, w_5, b):
    y_pred = []
    for i in range(len(x_train_2)):
        y_pred.append(w_2 * x_train_2[i] + w_5 * x_train_5[i] + b)
    return np.array(y_pred)

# Create a figure with two subplots
plt.figure(figsize=(12, 6))

# Plot learning curve
plt.subplot(1, 2, 1)
plt.plot(range(iterations), cost_history, color="blue")
plt.xlabel("Iterations")
plt.ylabel("Cost (MSE)")
plt.title("Learning Curve for Linear Regression")

#Print the graph and the lines
plt.subplot(1, 2, 2)
#this is the new x-value, which combines interest and interaction quests acquired
interactions["x-values"] = interactions["Interest"] + interactions["Interaction quests acquired"]
plt.scatter(interactions["x-values"], interactions["Interaction length"], alpha=0.5, color="orange")
plt.plot(interactions["x-values"], predict(x_train_2,x_train_5,w_optimal_2,w_optimal_5,b), color="blue", label="Best Fit Line")
plt.xlabel("Interest + Interaction quests acquired")
plt.ylabel("Interaction length")
plt.title("Interaction length vs (Interest + Interaction quests acquired)")

plt.tight_layout()
plt.show()

print("Interest w:")
print(w_optimal_2)
print("Interaction quests acquired w:")
print(w_optimal_5)