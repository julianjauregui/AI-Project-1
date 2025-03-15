#these are the following libraries that will be used for the project
#ensure that these packages are installed before hand by using pip install
import csv
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import cmocean
import os
import seaborn as sns


#This folder contains the datasets that we will be using for the project
import Datasets


#Interactions holds the data that happen between the players and the NPCs interactions
interactions = pd.read_csv(os.path.join("Datasets", "Interactions_Final_v10_Dataset.csv"))
#ogNPC holds the information of the original NPCs and their interactions
ogNPC = pd.read_csv(os.path.join("Datasets", "Updated_Old_NPCs.csv"))
#newNPC holds the information of the new, seasonal, NPCs and their interactions
newNPC = pd.read_csv(os.path.join("Datasets", "Updated_Seasonal_NPCs.csv"))


#this line ensures that all columns are shown for the pandas operations and summaries
pd.set_option('display.max_columns', None)

#prints a couple of sample lines and summarizes the interaction dataset
print("Sample Interactions: ")
print(interactions.head())
print("\n\n\n")
print("Interaction Summary: ")
print(interactions.describe())



# First, visualize raw relationships (without scaling)

# This style of plot uses multiple panels within the same figure
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.scatter(interactions["User level"], interactions["Interaction length"], alpha=0.5, color="crimson")
plt.xlabel("User level")
plt.ylabel("Interaction length")
plt.title("Interaction length vs User Level")

plt.subplot(2, 2, 2)
plt.scatter(interactions["NPC friendliness"], interactions["Interaction length"], alpha=0.5, color="darkorange")
plt.xlabel("NPC friendliness")
plt.ylabel("Interaction length")
plt.title("Interaction length vs. NPC friendliness")

plt.subplot(2, 2, 3)
plt.scatter(interactions["Interest"], interactions["Interaction length"], alpha=0.5, color="royalblue")
plt.xlabel("Interest")
plt.ylabel("Interaction length")
plt.title("Interaction length vs Interest")

plt.subplot(2, 2, 4)
plt.scatter(interactions["Interaction quests acquired"], interactions["Interaction length"], alpha=0.5, color="lightgreen")
plt.xlabel("Interaction quests acquired")
plt.ylabel("Interaction length")
plt.title("Interaction length vs Interaction quests acquired")


plt.tight_layout()
plt.show()