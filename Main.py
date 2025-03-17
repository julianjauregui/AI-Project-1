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


#This helps us visualize the interactions dataset
#the first category is the interest
sns.histplot(data=interactions, x="Interest")
plt.title("Distribution of Interest")
plt.show()
#the second category is the User Level
sns.histplot(data=interactions, x="User level")
plt.title("Distribution of User Level")
plt.show()
#the third category is the NPC Friendliness
sns.histplot(data=interactions, x="NPC friendliness")
plt.title("Distribution of NPC Friendliness")
plt.show()
#the fourth category is the Interaction length
sns.histplot(data=interactions, x="Interaction length")
plt.title("Distribution of Interaction length")
plt.show()
#the fifth category is the Interaction quests acquired
sns.histplot(data=interactions, x="Interaction quests acquired")
plt.title("Distribution of Interaction quests acquired")
plt.show()
plt.show()



# First, visualize raw relationships (without scaling)
#We will be making the target the interest
# This style of plot uses multiple panels within the same figure
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)

#this maps interest vs user level
plt.scatter(interactions["User level"], interactions["Interest"], alpha=0.5, color="crimson")
plt.xlabel("User level")
plt.ylabel("Interest")
plt.title("Interest vs User Level")

#this maps interest vs NPC Friendliness
plt.subplot(2, 2, 2)
plt.scatter(interactions["NPC friendliness"], interactions["Interest"], alpha=0.5, color="darkorange")
plt.xlabel("NPC friendliness")
plt.ylabel("Interest")
plt.title("Interest vs. NPC friendliness")

#this maps interest vs Interaction length
plt.subplot(2, 2, 3)
plt.scatter(interactions["Interaction length"], interactions["Interest"], alpha=0.5, color="royalblue")
plt.xlabel("Interaction length")
plt.ylabel("Interest")
plt.title("Interest vs. Interaction length")

#this maps interest vs interaction quests acquired
plt.subplot(2, 2, 4)
plt.scatter(interactions["Interaction quests acquired"], interactions["Interest"], alpha=0.5, color="lightgreen")
plt.xlabel("Interaction quests acquired")
plt.ylabel("Interest")
plt.title("Interest vs Interaction quests acquired")


plt.tight_layout()
plt.show()